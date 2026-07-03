# lawyer_chatbot
## Architecture

The assistant is a **RAG** (Retrieval-Augmented Generation) application:

1. `ingest.py` splits the Criminal Code (`docs/ukr_cc_ready.txt`) into one chunk per
   article (`Стаття N.`), embeds them with OpenAI embeddings and stores the vectors
   in an **embedded Qdrant** vector database that runs in-process and persists to a
   local folder (`qdrant_data/`) — no server or separate container required.
2. `chatbot.py` retrieves the most relevant articles for a question and asks the LLM
   to give a legal qualification grounded in that retrieved context.
3. `app.py` is a Streamlit chat UI on top of the chatbot.

```
docs/ ──ingest.py──▶ Qdrant (vectors) ──chatbot.py──▶ LLM ──app.py──▶ chat UI
```

| File | Purpose |
| --- | --- |
| `config.py` | Settings loaded from `.env` |
| `vectorstore.py` | Embedded Qdrant + embeddings helpers |
| `ingest.py` | Build/refresh the knowledge base |
| `chatbot.py` | RAG retrieval + answering chain |
| `app.py` | Streamlit chat interface |

Qdrant is embedded — it runs inside the Python process and stores its data in the
`qdrant_data/` folder. There is nothing separate to start.

## Quickstart (local)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # add your OPENAI_API_KEY

python ingest.py --recreate   # build the vector index into qdrant_data/
streamlit run app.py          # launch the chat UI
```

Open http://localhost:8501

> Note: the embedded Qdrant store can be opened by only one process at a time.
> Stop the Streamlit app before re-running `python ingest.py`, and vice versa.

## Quickstart (Docker)

```bash
cp .env.example .env          # add your OPENAI_API_KEY
docker compose up -d --build  # builds and starts just the app
# one-time: build the vector index inside the running app container
docker compose exec app python ingest.py --recreate
```

The vector index is persisted in the `qdrant_data` Docker volume.
