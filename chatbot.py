"""RAG chatbot: retrieve relevant articles from Qdrant and answer with the LLM."""
from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, AIMessageChunk
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from config import settings
from pricing import get_model_prices
from reranker import rerank
from vectorstore import get_client, get_vectorstore, latest_collection_name

SYSTEM_PROMPT = """Ти викладач кримінального права України.
Відповідай державною (українською) мовою, спираючись ВИКЛЮЧНО на наведені нижче \
витяги з Кримінального кодексу України (контекст).

Правила відповіді:
- Дай юридичну кваліфікацію діяння у форматі: \
"[Ім'я особи] — [частина, стаття КК України, назва статті]". Статей може бути декілька.
- Обов'язково посилайся на конкретні номери статей із контексту.
- Якщо у контексті немає відповіді, чесно скажи: \
"Не певен, але вважаю що" і обґрунтуй припущення. Не вигадуй неіснуючих статей.

Контекст:
{context}
"""

PROMPT = ChatPromptTemplate.from_messages(
    [("system", SYSTEM_PROMPT), ("human", "{question}")]
)


@dataclass
class TokenUsage:
    """Token accounting for one generator call, with a cost estimate.

    ``input_tokens`` is the full prompt size (as billed); ``cached_tokens`` is the
    subset of it served from the prompt cache at a reduced rate. ``cost_usd`` uses
    per-model prices resolved by name (see ``pricing.get_model_prices``), billing
    cached tokens at the cached rate and the rest at their own rates.
    """

    input_tokens: int = 0
    cached_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float = 0.0

    @classmethod
    def from_message(cls, message: AIMessage) -> "TokenUsage":
        usage = getattr(message, "usage_metadata", None) or {}
        input_tokens = usage.get("input_tokens", 0)
        output_tokens = usage.get("output_tokens", 0)
        cached_tokens = (usage.get("input_token_details") or {}).get("cache_read", 0)
        # Fresh (uncached) input is billed at the full input rate; cached input at
        # the discounted rate. Prices are per-token, resolved by model name.
        prices = get_model_prices(settings.LLM_MODEL)
        fresh_input = max(input_tokens - cached_tokens, 0)
        cost_usd = (
            fresh_input * prices.input
            + cached_tokens * prices.cached_input
            + output_tokens * prices.output
        )
        return cls(
            input_tokens=input_tokens,
            cached_tokens=cached_tokens,
            output_tokens=output_tokens,
            total_tokens=usage.get("total_tokens", input_tokens + output_tokens),
            cost_usd=round(cost_usd, 6),
        )

    def as_dict(self) -> dict:
        return {
            "input_tokens": self.input_tokens,
            "cached_tokens": self.cached_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "cost_usd": self.cost_usd,
        }


@dataclass
class Answer:
    text: str
    sources: list[Document]
    usage: TokenUsage = field(default_factory=TokenUsage)


@dataclass
class StreamingAnswer:
    """Answer that streams the LLM text as it is generated.

    ``sources`` are known up front (retrieval finishes before generation starts).
    Iterating the object yields text pieces (drive it with ``st.write_stream``);
    once iteration completes, ``text`` holds the full answer and ``usage`` the
    token accounting, both accumulated from the streamed chunks.
    """

    sources: list[Document]
    _chunks: Iterator[AIMessageChunk]
    text: str = ""
    usage: TokenUsage = field(default_factory=TokenUsage)

    def __iter__(self) -> Iterator[str]:
        merged: AIMessageChunk | None = None
        for chunk in self._chunks:
            merged = chunk if merged is None else merged + chunk
            if chunk.content:
                self.text += chunk.content
                yield chunk.content
        # The concatenated chunk carries the aggregated usage_metadata that
        # arrives (with stream_usage=True) in the final chunk of the stream.
        if merged is not None:
            self.usage = TokenUsage.from_message(merged)


def _format_context(docs: list[Document]) -> str:
    return "\n\n".join(
        f"[{d.metadata.get('title', d.metadata.get('article', 'джерело'))}]\n{d.page_content}"
        for d in docs
    )


def _dedupe_by_article(docs: list[Document]) -> list[Document]:
    """Keep at most one chunk per article, preserving order (best-ranked first).

    A long article is split into several chunks at ingest, all sharing the same
    ``article``/``title`` metadata, so vector search or reranking can surface
    multiple chunks of the same article — which reads as "the same source
    repeated". Since ``docs`` arrive ranked best-first, the first chunk seen for
    an article is its most relevant one; keep it and drop the rest. Falls back to
    the chunk text as the key (collapsing byte-identical chunks, e.g. from an
    accidental double ingest) when an article has no metadata.
    """
    seen: set[str] = set()
    unique: list[Document] = []
    for d in docs:
        key = d.metadata.get("article") or d.metadata.get("title") or d.page_content
        if key in seen:
            continue
        seen.add(key)
        unique.append(d)
    return unique


class LawyerChatbot:
    def __init__(self) -> None:
        self.llm = ChatOpenAI(
            model=settings.LLM_MODEL,
            temperature=settings.TEMPERATURE,
            api_key=settings.OPENAI_API_KEY,
            # Emit usage_metadata in the final streamed chunk so token/cost
            # accounting still works when we stream the answer.
            stream_usage=True,
        )
        # With reranking on, retrieve a larger candidate pool and let the
        # cross-encoder narrow it down. Otherwise over-fetch a small multiple of
        # TOP_K so that collapsing multi-chunk articles to one entry each (see
        # _dedupe_by_article) still leaves TOP_K distinct articles.
        self.rerank_enabled = settings.RERANK_ENABLED
        fetch_k = (
            settings.RERANK_CANDIDATES if self.rerank_enabled else settings.TOP_K * 4
        )
        # Bind to the most recent ingest; its timestamped name tells you when it
        # was built.
        self.collection_name = latest_collection_name(get_client())
        self.retriever = get_vectorstore(
            collection_name=self.collection_name
        ).as_retriever(search_kwargs={"k": fetch_k})
        # Keep the raw AIMessage (no StrOutputParser) so token usage metadata
        # survives the chain and can be accounted for.
        self.chain = PROMPT | self.llm

    def _retrieve(self, question: str) -> list[Document]:
        """Fetch, (optionally) rerank, dedupe by article, and keep TOP_K."""
        docs = self.retriever.invoke(question)
        if self.rerank_enabled:
            # Rank the whole candidate pool (don't truncate yet); dedupe and the
            # TOP_K cut happen below, after duplicate articles are collapsed.
            docs = rerank(question, docs, top_k=len(docs))
        docs = _dedupe_by_article(docs)
        return docs[: settings.TOP_K]

    def ask(self, question: str) -> Answer:
        docs = self._retrieve(question)
        message: AIMessage = self.chain.invoke(
            {"question": question, "context": _format_context(docs)}
        )
        return Answer(
            text=message.content,
            sources=docs,
            usage=TokenUsage.from_message(message),
        )

    def stream(self, question: str) -> StreamingAnswer:
        """Retrieve context, then stream the LLM answer token by token."""
        docs = self._retrieve(question)
        chunks = self.chain.stream(
            {"question": question, "context": _format_context(docs)}
        )
        return StreamingAnswer(sources=docs, _chunks=chunks)


if __name__ == "__main__":
    import sys

    bot = LawyerChatbot()
    query = " ".join(sys.argv[1:]) or "Яка стаття передбачає покарання за побої?"
    result = bot.ask(query)
    print(result.text)
    print("\n--- Джерела ---")
    for d in result.sources:
        print("-", d.metadata.get("title", d.metadata.get("article")))
    u = result.usage
    print(
        f"\n--- Токени --- вхід {u.input_tokens} (кеш {u.cached_tokens}) · "
        f"вихід {u.output_tokens} · усього {u.total_tokens} · ${u.cost_usd:.6f}"
    )
