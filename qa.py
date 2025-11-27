"""Conversational retrieval QA for Thai laws via OpenRouter."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import List

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma


EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

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


def load_chain_components(persist_dir: Path, model: str, top_k: int):
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
    return llm, retriever


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
    llm, retriever = load_chain_components(
        persist_dir=args.persist_dir,
        model=args.model,
        top_k=args.top_k,
    )

    chat_history: List[HumanMessage | AIMessage] = []

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

        condense_msgs = CONDENSE_PROMPT.format_messages(
            chat_history=chat_history,
            input=question,
        )
        condensed_question = llm.invoke(condense_msgs).content.strip()

        docs = retriever.invoke(condensed_question)
        context_text = "\n\n".join(doc.page_content for doc in docs)

        qa_msgs = QA_PROMPT.format_messages(
            context=context_text,
            input=condensed_question,
        )
        answer_msg = llm.invoke(qa_msgs)
        answer = answer_msg.content.strip()

        chat_history.extend(
            [HumanMessage(content=question), AIMessage(content=answer)]
        )

        print(f"\nตอบ: {answer}\n")

        if docs:
            print("แหล่งอ้างอิง:")
            for doc in docs:
                meta = doc.metadata or {}
                source = meta.get("source", "unknown source")
                page = meta.get("page")
                page_text = f" (page {page + 1})" if isinstance(page, int) else ""
                print(f"- {source}{page_text}")
            print()


if __name__ == "__main__":
    main()
