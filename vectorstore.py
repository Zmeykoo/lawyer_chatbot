"""Qdrant-backed vector store helpers shared by ingestion and the chatbot."""
from __future__ import annotations

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


def collection_exists(client: QdrantClient) -> bool:
    return client.collection_exists(settings.COLLECTION_NAME)


def recreate_collection(client: QdrantClient, vector_size: int) -> None:
    """(Re)create the collection with cosine distance, dropping any old data."""
    if client.collection_exists(settings.COLLECTION_NAME):
        client.delete_collection(settings.COLLECTION_NAME)
    client.create_collection(
        collection_name=settings.COLLECTION_NAME,
        vectors_config=qmodels.VectorParams(
            size=vector_size, distance=qmodels.Distance.COSINE
        ),
    )


def get_vectorstore(embeddings: Embeddings | None = None) -> QdrantVectorStore:
    """Return a vector store bound to the existing collection."""
    embeddings = embeddings or get_embeddings()
    return QdrantVectorStore(
        client=get_client(),
        collection_name=settings.COLLECTION_NAME,
        embedding=embeddings,
    )
