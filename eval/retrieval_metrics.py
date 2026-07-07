"""Classic information-retrieval metrics for the retrieval stage.

These score *which articles came back and in what order* against a ground-truth
set of relevant article numbers (``reference_articles`` in the golden set). They
use binary relevance (an article is relevant or not) and need no LLM — they are
cheap, deterministic, and the right tool for tuning ``TOP_K`` and reranking.

Conventions
-----------
``retrieved`` is the ranked list of article ids (best first, no duplicates — the
pipeline already collapses multi-chunk articles). ``relevant`` is the set of
ground-truth article ids. Ids are compared as normalized strings, so "126" and
"126-1" are (correctly) different articles.

  * Precision@k — of the top-k retrieved, the fraction that are relevant (÷ k).
  * Recall@k    — of the relevant articles, the fraction found in the top-k.
  * Hit Rate@k  — 1 if any relevant article is in the top-k, else 0.
  * MRR         — reciprocal rank (1/rank) of the first relevant article.
  * AP / MAP    — average precision per query; MAP is its mean over queries.
  * NDCG@k      — rank-discounted gain, normalized by the ideal ordering.
"""
from __future__ import annotations

import math
from collections.abc import Sequence


def normalize(article: object) -> str:
    """Canonical article id for comparison (e.g. ``"126"``, ``"368-2"``)."""
    return str(article).strip().lower()


def precision_at_k(retrieved: Sequence[str], relevant: set[str], k: int) -> float:
    top_k = retrieved[:k]
    if k <= 0:
        return 0.0
    return sum(1 for d in top_k if d in relevant) / k


def recall_at_k(retrieved: Sequence[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    return len(set(retrieved[:k]) & relevant) / len(relevant)


def hit_rate_at_k(retrieved: Sequence[str], relevant: set[str], k: int) -> float:
    return 1.0 if set(retrieved[:k]) & relevant else 0.0


def reciprocal_rank(retrieved: Sequence[str], relevant: set[str]) -> float:
    for rank, d in enumerate(retrieved, 1):
        if d in relevant:
            return 1.0 / rank
    return 0.0


def average_precision(retrieved: Sequence[str], relevant: set[str]) -> float:
    if not relevant:
        return 0.0
    hits = 0
    score = 0.0
    for rank, d in enumerate(retrieved, 1):
        if d in relevant:
            hits += 1
            score += hits / rank
    return score / len(relevant)


def _dcg(gains: Sequence[float]) -> float:
    # Position 1 is discounted by log2(2)=1, so the top hit is undiscounted.
    return sum(g / math.log2(rank + 1) for rank, g in enumerate(gains, 1))


def ndcg_at_k(retrieved: Sequence[str], relevant: set[str], k: int) -> float:
    gains = [1.0 if d in relevant else 0.0 for d in retrieved[:k]]
    ideal = _dcg([1.0] * min(len(relevant), k))
    if ideal == 0.0:
        return 0.0
    return _dcg(gains) / ideal


def evaluate_retrieval(
    per_query: list[tuple[Sequence[str], set[str]]], k_values: Sequence[int]
) -> dict[str, float]:
    """Average every metric over all queries.

    ``per_query`` is a list of ``(retrieved_ids, relevant_ids)`` pairs; queries
    with an empty relevant set are skipped (no ground truth to score against).
    Returns a flat ``{metric_name: mean_score}`` dict.
    """
    scored = [(r, rel) for r, rel in per_query if rel]
    if not scored:
        return {}
    n = len(scored)
    out: dict[str, float] = {"queries_scored": float(n)}
    for k in k_values:
        out[f"precision@{k}"] = sum(precision_at_k(r, rel, k) for r, rel in scored) / n
        out[f"recall@{k}"] = sum(recall_at_k(r, rel, k) for r, rel in scored) / n
        out[f"hit_rate@{k}"] = sum(hit_rate_at_k(r, rel, k) for r, rel in scored) / n
        out[f"ndcg@{k}"] = sum(ndcg_at_k(r, rel, k) for r, rel in scored) / n
    out["mrr"] = sum(reciprocal_rank(r, rel) for r, rel in scored) / n
    out["map"] = sum(average_precision(r, rel) for r, rel in scored) / n
    return out
