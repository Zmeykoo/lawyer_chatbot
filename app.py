"""Streamlit chat UI for the Ukrainian criminal-law RAG assistant."""
from __future__ import annotations

import streamlit as st

from chatbot import LawyerChatbot
from config import settings

st.set_page_config(page_title="Юридичний асистент (КК України)", page_icon="⚖️")


@st.cache_resource(show_spinner="Підключення до бази знань...")
def load_bot() -> LawyerChatbot:
    return LawyerChatbot()


def main() -> None:
    st.title("⚖️ Юридичний асистент")
    st.caption(
        f"RAG над Кримінальним кодексом України · модель {settings.LLM_MODEL} · "
        f"Qdrant `{settings.COLLECTION_NAME}`"
    )

    if not settings.OPENAI_API_KEY:
        st.error("OPENAI_API_KEY не задано. Додайте його у .env і перезапустіть.")
        st.stop()

    try:
        bot = load_bot()
    except Exception as exc:  # noqa: BLE001 - surface setup issues to the user
        st.error(
            "Не вдалося завантажити базу знань. "
            "Переконайтеся, що виконано `python ingest.py --recreate`.\n\n"
            f"Деталі: {exc}"
        )
        st.stop()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if question := st.chat_input("Опишіть ситуацію або задайте питання..."):
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Аналізую Кримінальний кодекс..."):
                answer = bot.ask(question)
            st.markdown(answer.text)
            with st.expander("Використані статті"):
                for doc in answer.sources:
                    st.markdown(
                        f"**{doc.metadata.get('title', doc.metadata.get('article', 'джерело'))}**"
                    )
                    st.caption(doc.page_content[:500] + "…")
        st.session_state.messages.append(
            {"role": "assistant", "content": answer.text}
        )


if __name__ == "__main__":
    main()
