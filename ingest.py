"""Build the local Qdrant knowledge base from the Criminal Code of Ukraine.

Usage:
    python ingest.py                # index the default SOURCE_FILE
    python ingest.py --recreate     # drop and rebuild the collection
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from tqdm import tqdm

from config import settings
from utils.file_handler import Reader
from vectorstore import get_client, get_embeddings, get_vectorstore, recreate_collection

# Matches an article header like "Стаття 126-1." at the start of a line.
ARTICLE_RE = re.compile(r"(?m)^(Стаття\s+[\d\-]+\..*)$")


def split_by_article(text: str) -> list[Document]:
    """Split the code into one Document per article, keeping the header as metadata."""
    matches = list(ARTICLE_RE.finditer(text))
    docs: list[Document] = []
    for i, match in enumerate(matches):
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[start:end].strip()
        header = match.group(1).strip()
        article_no = header.split(".")[0].replace("Стаття", "").strip()
        docs.append(
            Document(
                page_content=body,
                metadata={"article": article_no, "title": header},
            )
        )
    return docs


def load_documents() -> list[Document]:
    source = Path(settings.SOURCE_FILE)
    if not source.exists():
        raise FileNotFoundError(f"Source file not found: {source}")

    text = Reader.read_txt(str(source))
    docs = split_by_article(text)
    if not docs:  # fallback: no article headers found, treat as plain text
        docs = [Document(page_content=text, metadata={"source": str(source)})]

    # Split articles that are longer than the model-friendly chunk size.
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(docs)


def main() -> None:
    parser = argparse.ArgumentParser(description="Index legal documents into Qdrant.")
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Drop the existing collection before indexing.",
    )
    args = parser.parse_args()

    if settings.EMBEDDING_PROVIDER == "openai" and not settings.OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is not set (see .env.example).")

    print(f"Loading and chunking {settings.SOURCE_FILE} ...")
    chunks = load_documents()
    print(f"Prepared {len(chunks)} chunks.")

    embeddings = get_embeddings()
    client = get_client()

    if args.recreate or not client.collection_exists(settings.COLLECTION_NAME):
        vector_size = len(embeddings.embed_query("dimension probe"))
        print(f"Creating collection '{settings.COLLECTION_NAME}' (dim={vector_size}).")
        recreate_collection(client, vector_size)

    store = get_vectorstore(embeddings)
    batch_size = settings.INGEST_BATCH_SIZE
    for start in tqdm(
        range(0, len(chunks), batch_size), desc="Indexing", unit="batch"
    ):
        store.add_documents(chunks[start : start + batch_size])
    print(f"Indexed {len(chunks)} chunks into '{settings.COLLECTION_NAME}'. Done.")


if __name__ == "__main__":
    main()
