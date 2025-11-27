"""Conversational retrieval QA for Thai laws via OpenRouter."""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from dotenv import load_dotenv

from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter


EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


CONDENSE_PROMPT = PromptTemplate.from_template(
    "ให้เรียบเรียงคำถามใหม่โดยอ้างอิงจากบทสนทนาก่อนหน้า "
    "ให้เป็นประโยคคำถามที่สมบูรณ์และเข้าใจได้ด้วยตัวเอง (Standalone Question) เป็นภาษาไทย\n"
    "ประวัติการสนทนา:\n{chat_history}\n"
    "คำถามปัจจุบัน: {question}\n"
    "คำถามแบบ Standalone:"
)

QA_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template=(
        "You are a Thai Law Expert Assistant.\n"
        "Use the provided context to answer.\n"
        "If unsure, say 'ไม่พบข้อมูล'.\n"
        "Always cite the Section (มาตรา) number.\n\n"
        "Context:\n{context}\n\n"
        "Question:\n{question}\n"
        "Answer in Thai."
    ),
)


def create_chain(persist_dir: Path, model: str, top_k: int) -> ConversationalRetrievalChain:
    load_dotenv()
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vectordb = Chroma(
        persist_directory=str(persist_dir),
        embedding_function=embeddings,
    )
    retriever = vectordb.as_retriever(search_kwargs={"k": top_k})

    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENROUTER_API_KEY is required.")

    llm = ChatOpenAI(
        api_key=api_key,
        base_url=OPENROUTER_BASE_URL,
        model=model,
        temperature=0,
    )

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
    )

    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        condense_question_prompt=CONDENSE_PROMPT,
        combine_docs_chain_kwargs={"prompt": QA_PROMPT},
        return_source_documents=True,
    )
    return chain


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Thai Law conversational QA using OpenRouter + Chroma."
    )
    parser.add_argument(
        "--persist-dir",
        type=Path,
        default=Path("vectorstore"),
        help="Path to persisted Chroma database.",
    )
    parser.add_argument(
        "--model",
        default="openai/gpt-4o-mini",
        help="OpenRouter model identifier (e.g., openai/gpt-4o-mini or google/gemini-flash-1.5).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=4,
        help="Number of retrieved chunks to pass to the LLM.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    chain = create_chain(
        persist_dir=args.persist_dir,
        model=args.model,
        top_k=args.top_k,
    )

    print("Thai Law RAG QA (OpenRouter). Type 'exit' to quit.")
    while True:
        try:
            question = input("ถาม: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nจบการทำงาน.")
            break

        if question.lower() in {"exit", "quit"}:
            print("จบการทำงาน.")
            break
        if not question:
            continue

        result = chain.invoke({"question": question})
        answer = result.get("answer", "").strip()

        print(f"\nตอบ: {answer}\n")

        sources = result.get("source_documents") or []
        if sources:
            print("แหล่งอ้างอิง:")
            for doc in sources:
                meta = doc.metadata or {}
                source = meta.get("source", "unknown source")
                page = meta.get("page")
                page_text = f" (page {page + 1})" if isinstance(page, int) else ""
                print(f"- {source}{page_text}")
            print()


if __name__ == "__main__":
    main()
