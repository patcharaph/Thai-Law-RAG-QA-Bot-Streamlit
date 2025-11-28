"""Streamlit UI for Thai Law RAG QA using OpenRouter + Chroma."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, List

import streamlit as st
from dotenv import find_dotenv, load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI


EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MODEL = "openai/gpt-4o-mini"
DEFAULT_TOP_K = 4
DEFAULT_PERSIST_DIR = Path("vectorstore")

CONDENSE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "ให้เรียบเรียงคำถามใหม่โดยอ้างอิงจากบทสนทนาก่อนหน้า "
            "ให้เป็นประโยคคำถามที่สมบูรณ์และเข้าใจได้ด้วยตัวเอง (Standalone Question) เป็นภาษาไทย",
        ),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

QA_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a Thai Law Expert Assistant.\n"
            "Use the provided context to answer.\n"
            "If unsure, say 'ไม่พบข้อมูล'.\n"
            "Always cite the Section (มาตรา) number.\n"
            "Answer in Thai.",
        ),
        ("system", "Context:\n{context}"),
        ("human", "{input}"),
    ]
)


def set_cyberpunk_theme() -> None:
    """Inject a simple cyberpunk-inspired theme."""
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700&family=Share+Tech+Mono&display=swap');
        :root {
            --neo-pink: #ff2fd0;
            --neo-cyan: #00f6ff;
            --neo-yellow: #f2ff49;
            --panel: rgba(10, 10, 20, 0.75);
        }
        body, [data-testid="stAppViewContainer"] {
            background: radial-gradient(circle at 20% 20%, rgba(0, 246, 255, 0.08), transparent 25%),
                        radial-gradient(circle at 80% 10%, rgba(255, 47, 208, 0.09), transparent 25%),
                        linear-gradient(135deg, #0a0b15 0%, #0c1024 55%, #06060d 100%);
            color: #e8f7ff;
            font-family: 'Share Tech Mono', monospace;
        }
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, rgba(12,16,36,0.95), rgba(12,16,36,0.8));
            border-right: 1px solid rgba(0, 246, 255, 0.35);
        }
        h1, h2, h3, .stMarkdown h1, .stMarkdown h2 {
            font-family: 'Orbitron', 'Share Tech Mono', monospace;
            letter-spacing: 1px;
            color: var(--neo-cyan);
            text-shadow: 0 0 18px rgba(0, 246, 255, 0.4);
        }
        .stButton>button, .stSelectbox>div>div, .stSlider>div>div>div>div {
            font-family: 'Orbitron', 'Share Tech Mono', monospace;
        }
        .stButton>button {
            background: linear-gradient(90deg, var(--neo-pink), var(--neo-cyan));
            color: #0a0b15;
            border: none;
            box-shadow: 0 0 12px rgba(0, 246, 255, 0.45), 0 0 20px rgba(255, 47, 208, 0.35);
        }
        .stButton>button:hover {
            filter: brightness(1.08);
        }
        .stTextInput>div>div>input, .stSelectbox>div>div>div {
            background: rgba(255,255,255,0.05) !important;
            border: 1px solid rgba(0, 246, 255, 0.35);
            color: #e8f7ff !important;
        }
        .stSlider [data-baseweb="slider"]>div {
            background: rgba(255, 47, 208, 0.4);
        }
        .stSlider [data-baseweb="slider"]>div>div {
            background: rgba(0, 246, 255, 0.8);
        }
        .stChatMessage {
            background: var(--panel);
            border: 1px solid rgba(0, 246, 255, 0.18);
            box-shadow: 0 0 8px rgba(0, 246, 255, 0.12);
        }
        code, pre {
            background: rgba(255,255,255,0.05);
            border: 1px solid rgba(0, 246, 255, 0.25);
        }
        a { color: var(--neo-yellow); }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource(show_spinner=False)
def load_vectordb(persist_dir: Path) -> Chroma:
    """Load the persistent Chroma DB once."""
    load_dotenv(find_dotenv(), override=True)
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    return Chroma(persist_directory=str(persist_dir), embedding_function=embeddings)


@st.cache_resource(show_spinner=False)
def load_llm(model: str, api_key: str) -> ChatOpenAI:
    """Instantiate the LLM once per model."""
    return ChatOpenAI(
        api_key=api_key,
        base_url=OPENROUTER_BASE_URL,
        model=model,
        temperature=0,
    )


def ensure_api_key() -> str | None:
    """Return API key if available, otherwise warn the user."""
    load_dotenv(find_dotenv(), override=True)
    api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        st.warning(
            "ยังไม่พบค่า `OPENROUTER_API_KEY` กรุณาสร้างไฟล์ `.env` แล้วกำหนดคีย์ "
            "หรือส่งผ่าน Environment Variable ก่อนเริ่มใช้งาน"
        )
        return None
    return api_key


def format_docs(docs: Iterable[Document]) -> str:
    """Readable source listing for display."""
    lines = []
    for doc in docs:
        meta = doc.metadata or {}
        source = meta.get("source", "unknown source")
        page = meta.get("page")
        page_text = f" (page {page + 1})" if isinstance(page, int) else ""
        lines.append(f"- {source}{page_text}")
    return "\n".join(lines)


def to_langchain_history(messages: List[dict]) -> List[HumanMessage | AIMessage]:
    """Convert session messages to LangChain chat history."""
    converted: List[HumanMessage | AIMessage] = []
    for msg in messages:
        if msg["role"] == "user":
            converted.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            converted.append(AIMessage(content=msg["content"]))
    return converted


def main() -> None:
    st.set_page_config(page_title="Thai Law RAG QA", page_icon="⚖️", layout="wide")
    set_cyberpunk_theme()
    st.title("กฎหมายแรงงานไทย Q&A")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    with st.sidebar:
        st.header("ตั้งค่า")
        model = st.selectbox(
            "เลือกโมเดล",
            options=["openai/gpt-4o-mini", "google/gemini-flash-1.5"],
            index=0,
        )
        top_k = st.slider("Top-K เอกสารที่ดึงมา", min_value=1, max_value=8, value=DEFAULT_TOP_K)
        persist_dir = st.text_input(
            "โฟลเดอร์ ChromaDB",
            value=str(DEFAULT_PERSIST_DIR),
            help="ตำแหน่งฐานข้อมูลที่สร้างด้วย build_kb.py",
        )
        if st.button("Clear Chat", type="secondary"):
            st.session_state.messages = []
            st.rerun()

    api_key = ensure_api_key()
    if not api_key:
        return

    persist_path = Path(persist_dir)
    if not persist_path.exists():
        st.warning(f"ไม่พบโฟลเดอร์ฐานข้อมูล: {persist_path}. กรุณารัน build_kb.py ก่อน")
        return

    try:
        vectordb = load_vectordb(persist_path)
        llm = load_llm(model=model, api_key=api_key)
    except Exception as exc:  # noqa: BLE001
        st.error(f"โหลดทรัพยากรไม่สำเร็จ: {exc}")
        return

    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    user_input = st.chat_input("พิมพ์คำถามเกี่ยวกับกฎหมายไทยที่นี่")
    if not user_input:
        return

    st.chat_message("user").write(user_input)

    with st.chat_message("assistant"):
        with st.spinner("กำลังค้นหาคำตอบ..."):
            try:
                chat_history = to_langchain_history(st.session_state.messages)
                condense_msgs = CONDENSE_PROMPT.format_messages(
                    chat_history=chat_history,
                    input=user_input,
                )
                condensed_question = llm.invoke(condense_msgs).content.strip()

                retriever = vectordb.as_retriever(search_kwargs={"k": top_k})
                docs = retriever.invoke(condensed_question)
                context_text = "\n\n".join(doc.page_content for doc in docs)

                qa_msgs = QA_PROMPT.format_messages(
                    context=context_text,
                    input=condensed_question,
                )
                answer_msg = llm.invoke(qa_msgs)
                answer = answer_msg.content.strip()

                st.write(answer)
                if docs:
                    st.caption("แหล่งอ้างอิง:")
                    st.markdown(format_docs(docs))

                st.session_state.messages.append({"role": "user", "content": user_input})
                st.session_state.messages.append({"role": "assistant", "content": answer})
            except Exception as exc:  # noqa: BLE001
                st.error(f"เกิดข้อผิดพลาดระหว่างประมวลผล: {exc}")


if __name__ == "__main__":
    main()
