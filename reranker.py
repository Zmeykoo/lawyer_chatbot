"""Cross-encoder reranking for retrieved chunks.

The vector store retrieves candidates by embedding similarity (a bi-encoder),
which is fast but coarse. A cross-encoder scores each (query, chunk) pair jointly
and is far more accurate at ordering, so we use it to re-rank a larger candidate
pool down to the final TOP_K. Gated by ``RERANK_ENABLED`` in the config.
"""
from __future__ import annotations

from langchain_core.documents import Document

from config import settings

_reranker = None


def get_reranker():
    """Return a process-wide, lazily loaded CrossEncoder.

    Imported and instantiated on first use so that runs with reranking disabled
    never pay the model-load cost or need sentence-transformers loaded.
    """
    global _reranker
    if _reranker is None:
        from sentence_transformers import CrossEncoder

        _reranker = CrossEncoder(settings.RERANK_MODEL, device=settings.RERANK_DEVICE)
    return _reranker


def rerank(
    query: str, docs: list[Document], top_k: int | None = None
) -> list[Document]:
    """Re-order ``docs`` by cross-encoder relevance to ``query``, keep ``top_k``."""
    if not docs:
        return docs

    top_k = settings.TOP_K if top_k is None else top_k
    model = get_reranker()
    scores = model.predict([(query, d.page_content) for d in docs])
    ranked = sorted(zip(scores, docs), key=lambda pair: pair[0], reverse=True)
    return [doc for _, doc in ranked[:top_k]]
