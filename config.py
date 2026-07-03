"""Central configuration loaded from environment variables (.env)."""
import os

from dotenv import load_dotenv

load_dotenv()


class Settings:
    # --- OpenAI ---
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    LLM_MODEL: str = os.getenv("LLM_MODEL", "gpt-4o")
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    TEMPERATURE: float = float(os.getenv("TEMPERATURE", "0.0"))

    # --- Embeddings ---
    # Which encoder backend to use: "openai" or "huggingface".
    EMBEDDING_PROVIDER: str = os.getenv("EMBEDDING_PROVIDER", "openai").lower()

    # --- Hugging Face encoders (transformers / sentence-transformers) ---
    # Any model id from the Hub, e.g. "intfloat/multilingual-e5-base".
    HF_EMBEDDING_MODEL: str = os.getenv(
        "HF_EMBEDDING_MODEL", "intfloat/multilingual-e5-base"
    )
    # "cpu", "cuda", "cuda:0", "mps", ...
    HF_DEVICE: str = os.getenv("HF_DEVICE", "cpu")
    # Normalize vectors to unit length (recommended for cosine distance).
    HF_NORMALIZE_EMBEDDINGS: bool = (
        os.getenv("HF_NORMALIZE_EMBEDDINGS", "true").lower() in ("1", "true", "yes")
    )

    # --- Qdrant (embedded, on-disk vector DB) ---
    # Qdrant runs in-process and persists to this local directory — no server
    # or separate container required.
    QDRANT_PATH: str = os.getenv("QDRANT_PATH", "qdrant_data")
    COLLECTION_NAME: str = os.getenv("COLLECTION_NAME", "ukr_criminal_code")

    # --- Ingestion / retrieval ---
    DOCS_DIR: str = os.getenv("DOCS_DIR", "docs")
    # File(s) that are already cleaned and split by "Стаття N."
    SOURCE_FILE: str = os.getenv("SOURCE_FILE", "docs/ukr_cc_ready.txt")
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "1200"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "150"))
    TOP_K: int = int(os.getenv("TOP_K", "5"))
    # Chunks embedded/upserted per batch during ingestion (progress granularity).
    INGEST_BATCH_SIZE: int = int(os.getenv("INGEST_BATCH_SIZE", "64"))


settings = Settings()
