"""Build a Chroma vector store from Thai law PDF files."""
from __future__ import annotations

import argparse
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter


SEPARATORS = ["\n\n", "มาตรา", "\n", " ", ""]
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def load_pdfs(pdf_dir: Path) -> list:
    pdf_paths = sorted(pdf_dir.rglob("*.pdf"))
    if not pdf_paths:
        raise FileNotFoundError(f"No PDF files found under {pdf_dir}")

    documents = []
    for path in pdf_paths:
        loader = PyPDFLoader(str(path))
        documents.extend(loader.load())
    return documents


def build_vector_store(
    pdf_dir: Path,
    persist_dir: Path,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> Chroma:
    docs = load_pdfs(pdf_dir)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=SEPARATORS,
    )
    chunks = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    persist_dir.mkdir(parents=True, exist_ok=True)
    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding_function=embeddings,
        persist_directory=str(persist_dir),
        collection_name="thai_law",
    )
    vectordb.persist()
    return vectordb


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Chroma DB from Thai law PDFs.")
    parser.add_argument(
        "--pdf-dir",
        type=Path,
        default=Path("documents"),
        help="Directory containing PDF files.",
    )
    parser.add_argument(
        "--persist-dir",
        type=Path,
        default=Path("vectorstore"),
        help="Directory to store the Chroma database.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1000,
        help="Maximum characters per chunk.",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=200,
        help="Overlap between chunks.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    vectordb = build_vector_store(
        pdf_dir=args.pdf_dir,
        persist_dir=args.persist_dir,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )
    print(
        f"Chroma DB built at {args.persist_dir} with {vectordb._collection.count()} chunks."
    )


if __name__ == "__main__":
    main()
