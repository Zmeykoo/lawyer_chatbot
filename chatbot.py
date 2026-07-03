"""RAG chatbot: retrieve relevant articles from Qdrant and answer with the LLM."""
from __future__ import annotations

from dataclasses import dataclass

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from config import settings
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
class Answer:
    text: str
    sources: list[Document]


def _format_context(docs: list[Document]) -> str:
    return "\n\n".join(
        f"[{d.metadata.get('title', d.metadata.get('article', 'джерело'))}]\n{d.page_content}"
        for d in docs
    )


class LawyerChatbot:
    def __init__(self) -> None:
        self.llm = ChatOpenAI(
            model=settings.LLM_MODEL,
            temperature=settings.TEMPERATURE,
            api_key=settings.OPENAI_API_KEY,
        )
        # With reranking on, retrieve a larger candidate pool and let the
        # cross-encoder narrow it down to TOP_K; otherwise fetch TOP_K directly.
        self.rerank_enabled = settings.RERANK_ENABLED
        fetch_k = settings.RERANK_CANDIDATES if self.rerank_enabled else settings.TOP_K
        # Bind to the most recent ingest; its timestamped name tells you when it
        # was built.
        self.collection_name = latest_collection_name(get_client())
        self.retriever = get_vectorstore(
            collection_name=self.collection_name
        ).as_retriever(search_kwargs={"k": fetch_k})
        self.chain = PROMPT | self.llm | StrOutputParser()

    def ask(self, question: str) -> Answer:
        docs = self.retriever.invoke(question)
        if self.rerank_enabled:
            docs = rerank(question, docs, top_k=settings.TOP_K)
        text = self.chain.invoke(
            {"question": question, "context": _format_context(docs)}
        )
        return Answer(text=text, sources=docs)


if __name__ == "__main__":
    import sys

    bot = LawyerChatbot()
    query = " ".join(sys.argv[1:]) or "Яка стаття передбачає покарання за побої?"
    result = bot.ask(query)
    print(result.text)
    print("\n--- Джерела ---")
    for d in result.sources:
        print("-", d.metadata.get("title", d.metadata.get("article")))
