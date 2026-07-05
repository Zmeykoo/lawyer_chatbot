"""Offline RAGAS evaluation of the RAG pipeline.

Runs a fixed golden set of questions through ``LawyerChatbot`` and scores the
answers with RAGAS. Meant to be run by hand whenever retrieval, reranking, or the
model changes — not in the app request path.

    python eval/run_eval.py                       # score the default golden set
    python eval/run_eval.py --dataset eval/golden.jsonl --out eval/results.csv
    RERANK_ENABLED=true python eval/run_eval.py    # score with reranking on

IMPORTANT: this opens the embedded Qdrant store, which allows only one process at
a time — make sure ``app.py`` (and any other reader) is stopped first.

Metrics
-------
Retrieval / ranking (deterministic, no LLM; need ``reference_articles``):
  * precision@k, recall@k, hit_rate@k, ndcg@k, MRR, MAP — see retrieval_metrics.py.

Answer quality via RAGAS — reference-free (always run; need no ground truth):
  * faithfulness       — is every claim in the answer grounded in retrieved context?
  * answer_relevancy   — does the answer actually address the question?

Answer quality via RAGAS — reference-based (run only when every row has ``reference``):
  * context_precision  — are the retrieved articles the ones the reference needs?
  * context_recall     — did retrieval surface everything the reference needs?

The RAGAS judge scores in Ukrainian but its prompts are English-tuned, so read the
numbers as *relative* (config A vs B), not absolute grades. Use a strong judge.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow running as `python eval/run_eval.py` from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_openai import ChatOpenAI  # noqa: E402

from chatbot import LawyerChatbot  # noqa: E402
from config import settings  # noqa: E402
from eval.retrieval_metrics import evaluate_retrieval, normalize  # noqa: E402
from vectorstore import get_embeddings  # noqa: E402


def load_golden(path: Path) -> list[dict]:
    """Read the golden set (one JSON object per line)."""
    rows: list[dict] = []
    with path.open(encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{path}:{lineno}: bad JSON — {exc}") from exc
    if not rows:
        raise SystemExit(f"{path}: no rows to evaluate.")
    return rows


def run_bot(rows: list[dict], bot: LawyerChatbot) -> tuple[list, list, bool]:
    """Run each question through the bot once, in a single pass.

    Returns RAGAS samples, per-query ``(retrieved_article_ids, relevant_ids)``
    pairs for the IR metrics, and whether every row carried a ``reference``
    (which gates the RAGAS reference-based metrics).
    """
    from ragas import SingleTurnSample

    samples: list[SingleTurnSample] = []
    retrieval: list[tuple[list[str], set[str]]] = []
    have_all_refs = True
    for i, row in enumerate(rows, 1):
        question = row["question"]
        print(f"[{i}/{len(rows)}] {question[:70]}…", flush=True)
        answer = bot.ask(question)

        reference = (row.get("reference") or "").strip()
        have_all_refs = have_all_refs and bool(reference)
        samples.append(
            SingleTurnSample(
                user_input=question,
                response=answer.text,
                retrieved_contexts=[d.page_content for d in answer.sources],
                reference=reference or None,
            )
        )

        # Ranked article ids as returned (already deduped best-first, ≤ TOP_K)
        # vs the ground-truth relevant set for the IR metrics.
        retrieved_ids = [
            normalize(d.metadata.get("article"))
            for d in answer.sources
            if d.metadata.get("article") is not None
        ]
        relevant_ids = {normalize(a) for a in row.get("reference_articles") or []}
        retrieval.append((retrieved_ids, relevant_ids))

    return samples, retrieval, have_all_refs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path(__file__).with_name("golden.jsonl"),
        help="JSONL golden set (question / reference / reference_articles).",
    )
    parser.add_argument(
        "--judge-model",
        default="gpt-5.4-mini",
        help="LLM that scores the answers (use a strong one; default gpt-4o).",
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="Only evaluate the first N rows."
    )
    parser.add_argument(
        "--k-values",
        default="1,3,5",
        help="Comma-separated k cutoffs for precision/recall/hit_rate/ndcg (default 1,3,5).",
    )
    parser.add_argument(
        "--no-ragas",
        action="store_true",
        help="Skip the RAGAS answer-quality metrics; run only the IR retrieval metrics.",
    )
    parser.add_argument(
        "--out", type=Path, default=None, help="Optional CSV path for per-row scores."
    )
    args = parser.parse_args()

    k_values = [int(k) for k in args.k_values.split(",") if k.strip()]

    rows = load_golden(args.dataset)
    if args.limit:
        rows = rows[: args.limit]

    bot = LawyerChatbot()
    print(
        f"Config: collection={bot.collection_name} · llm={settings.LLM_MODEL} · "
        f"rerank={'on' if bot.rerank_enabled else 'off'} · top_k={settings.TOP_K}\n"
    )
    samples, retrieval, have_all_refs = run_bot(rows, bot)

    # --- Retrieval / ranking metrics (deterministic, no LLM) ---
    ir_scores = evaluate_retrieval(retrieval, k_values)
    print("\n=== Retrieval metrics ===")
    if not ir_scores:
        print("(no rows had `reference_articles` — nothing to score)")
    else:
        scored = int(ir_scores.pop("queries_scored"))
        print(f"(over {scored}/{len(rows)} rows with ground-truth articles)")
        for name, value in ir_scores.items():
            print(f"  {name:<14} {value:.4f}")

    if args.no_ragas:
        return

    # --- Answer-quality metrics (RAGAS, LLM judge) ---
    from ragas import EvaluationDataset, evaluate
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import (
        Faithfulness,
        LLMContextPrecisionWithReference,
        LLMContextRecall,
        ResponseRelevancy,
    )

    # Wrap the app's own embeddings so answer_relevancy uses the same encoder the
    # pipeline retrieves with; a separate strong model does the judging.
    judge = LangchainLLMWrapper(
        ChatOpenAI(model=args.judge_model, temperature=0.0, api_key=settings.OPENAI_API_KEY)
    )
    embeddings = LangchainEmbeddingsWrapper(get_embeddings())

    metrics = [Faithfulness(), ResponseRelevancy()]
    if have_all_refs:
        metrics += [LLMContextPrecisionWithReference(), LLMContextRecall()]
    else:
        print(
            "\nNote: some rows lack a `reference` — running reference-free RAGAS "
            "metrics only (add ground-truth answers to enable context precision/recall)."
        )

    result = evaluate(
        EvaluationDataset(samples=samples),
        metrics=metrics,
        llm=judge,
        embeddings=embeddings,
    )

    print("\n=== RAGAS scores ===")
    print(result)
    if args.out:
        result.to_pandas().to_csv(args.out, index=False)
        print(f"\nPer-row scores written to {args.out}")


if __name__ == "__main__":
    main()
