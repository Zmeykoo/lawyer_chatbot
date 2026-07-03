"""Qdrant-backed vector store helpers shared by ingestion and the chatbot."""
from __future__ import annotations

from datetime import datetime, timezone

from langchain_core.embeddings import Embeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from config import settings


def get_embeddings() -> Embeddings:
    """Build the embeddings backend selected by ``EMBEDDING_PROVIDER``."""
    provider = settings.EMBEDDING_PROVIDER

    if provider == "openai":
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(
            model=settings.EMBEDDING_MODEL,
            api_key=settings.OPENAI_API_KEY,
        )

    if provider in ("huggingface", "hf", "transformers"):
        from langchain_huggingface import HuggingFaceEmbeddings

        return HuggingFaceEmbeddings(
            model_name=settings.HF_EMBEDDING_MODEL,
            model_kwargs={
                "device": settings.HF_DEVICE,
                "trust_remote_code": True,
            },
            encode_kwargs={"normalize_embeddings": settings.HF_NORMALIZE_EMBEDDINGS},
        )

    raise ValueError(
        f"Unknown EMBEDDING_PROVIDER: {provider!r} (expected 'openai' or 'huggingface')."
    )


_client: QdrantClient | None = None


def get_client() -> QdrantClient:
    """Return a process-wide embedded Qdrant client persisting to ``QDRANT_PATH``.

    Local (on-disk) Qdrant allows only one client per storage folder within a
    process, so the client is cached and shared across ingestion and retrieval.
    """
    global _client
    if _client is None:
        _client = QdrantClient(path=settings.QDRANT_PATH)
    return _client


def new_collection_name() -> str:
    """A timestamped collection name so you can tell when an ingest was built.

    ``COLLECTION_NAME`` is treated as a base name; each ingest appends a UTC
    timestamp, e.g. ``ukr_criminal_code_20260703_142530``.
    """
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return f"{settings.COLLECTION_NAME}_{stamp}"


def latest_collection_name(client: QdrantClient) -> str | None:
    """Return the newest collection built from the configured base name.

    Timestamped names (``{COLLECTION_NAME}_{stamp}``) sort lexicographically in
    chronological order, so the greatest name is the most recent ingest. Falls
    back to a bare ``COLLECTION_NAME`` collection (older layout) if present, or
    ``None`` when nothing has been ingested yet.
    """
    prefix = f"{settings.COLLECTION_NAME}_"
    names = [c.name for c in client.get_collections().collections]
    stamped = sorted(n for n in names if n.startswith(prefix))
    if stamped:
        return stamped[-1]
    if settings.COLLECTION_NAME in names:
        return settings.COLLECTION_NAME
    return None


def recreate_collection(
    client: QdrantClient, vector_size: int, collection_name: str
) -> None:
    """(Re)create ``collection_name`` with cosine distance, dropping any old data."""
    if client.collection_exists(collection_name):
        client.delete_collection(collection_name)
    client.create_collection(
        collection_name=collection_name,
        vectors_config=qmodels.VectorParams(
            size=vector_size, distance=qmodels.Distance.COSINE
        ),
    )


def get_vectorstore(
    embeddings: Embeddings | None = None, collection_name: str | None = None
) -> QdrantVectorStore:
    """Return a vector store bound to a collection.

    Without an explicit ``collection_name``, the most recent ingest is used.
    """
    embeddings = embeddings or get_embeddings()
    client = get_client()
    collection_name = collection_name or latest_collection_name(client)
    if collection_name is None:
        raise RuntimeError(
            f"No '{settings.COLLECTION_NAME}' collection found. "
            "Run `python ingest.py --recreate` first."
        )
    return QdrantVectorStore(
        client=client,
        collection_name=collection_name,
        embedding=embeddings,
    )
