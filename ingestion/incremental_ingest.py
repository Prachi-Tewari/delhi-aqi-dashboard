"""Incremental RAG ingestion — add new documents without rebuilding everything.

Usage:
  # From Python
  from ingestion.incremental_ingest import add_documents, add_text_snippet

  add_documents("/path/to/new/files")           # add all .txt/.pdf in folder
  add_text_snippet("Some new fact...", "my_note")  # add a text snippet directly

  # From CLI
  python -m ingestion.incremental_ingest --add-dir data/pdfs/new_docs
  python -m ingestion.incremental_ingest --add-text "Delhi AQI hit 999 in Nov 2023" --name "news_update"
"""

import os
import json
import numpy as np
from pathlib import Path

# Paths (relative to project root)
PDF_DIR = "data/pdfs"
VECTOR_STORE_DIR = "embeddings/vector_store"
EMBED_MODEL = "BAAI/bge-base-en-v1.5"


def _get_sbert(model_name: str = EMBED_MODEL):
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(model_name)


def _chunk_text(text: str, doc_name: str) -> list:
    """Split text into semantic chunks with metadata."""
    from ingestion.chunk_docs import semantic_chunk, _infer_topic, _extract_year, _credibility_score, _approx_tokens
    raw_chunks = semantic_chunk(text, max_tokens=512, overlap_tokens=200)
    chunks = []
    for ci, chunk_text in enumerate(raw_chunks):
        chunks.append({
            "doc_name": doc_name,
            "page": ci + 1,
            "source_type": "text",
            "text": chunk_text,
            "topic": _infer_topic(chunk_text),
            "year": _extract_year(chunk_text),
            "credibility_score": _credibility_score(doc_name, "text"),
            "token_count": _approx_tokens(chunk_text),
        })
    return chunks


def _load_existing_store(vs_dir: str):
    """Load existing FAISS index and metadata, or return None."""
    from embeddings.vector_store import VectorStore
    info_path = os.path.join(vs_dir, "embed_info.json")
    if not os.path.exists(info_path):
        return None, None
    with open(info_path) as f:
        info = json.load(f)
    dim = info.get("dim", 384)
    try:
        vs = VectorStore.load(dim, vs_dir)
        return vs, info
    except Exception:
        return None, info


def _existing_doc_names(vs_dir: str) -> set:
    """Return set of doc_name values already in the store."""
    meta_path = os.path.join(vs_dir, "metadatas.json")
    if not os.path.exists(meta_path):
        return set()
    with open(meta_path) as f:
        metas = json.load(f)
    return {m.get("doc_name", "") for m in metas}


def add_documents(
    docs_dir: str = PDF_DIR,
    vs_dir: str = VECTOR_STORE_DIR,
    model_name: str = EMBED_MODEL,
    force: bool = False,
):
    """Scan docs_dir for new .txt/.pdf files not yet in the vector store.
    
    Embeds only the NEW files and appends them to the existing index.
    Set force=True to re-embed everything.
    """
    from ingestion.load_pdfs import load_all_pdfs
    from ingestion.chunk_docs import chunk_texts
    from embeddings.vector_store import VectorStore

    # Find which docs are already indexed
    existing_names = set() if force else _existing_doc_names(vs_dir)

    # Load all docs from directory
    all_pages = load_all_pdfs(docs_dir)

    # Filter to only new documents
    new_pages = [p for p in all_pages if p.get("doc_name", "") not in existing_names]
    if not new_pages:
        print("No new documents to add.")
        return 0

    new_doc_names = {p["doc_name"] for p in new_pages}
    print(f"Found {len(new_pages)} pages from {len(new_doc_names)} new documents: {new_doc_names}")

    # Chunk
    chunks = chunk_texts(new_pages)
    texts = [c["text"] for c in chunks]
    metas = [
        {"doc_name": c.get("doc_name"), "page": c.get("page"),
         "source_type": c.get("source_type"), "text": c.get("text")}
        for c in chunks
    ]

    # Embed
    model = _get_sbert(model_name)
    print(f"Embedding {len(texts)} new chunks …")
    X = model.encode(texts, show_progress_bar=True, convert_to_numpy=True).astype("float32")
    dim = X.shape[1]

    # Load or create vector store
    vs, info = _load_existing_store(vs_dir)
    if vs is None or force:
        # Full rebuild if needed
        if not force:
            # Also embed existing docs if no store exists
            all_chunks = chunk_texts(all_pages)
            all_texts = [c["text"] for c in all_chunks]
            all_metas = [
                {"doc_name": c.get("doc_name"), "page": c.get("page"),
                 "source_type": c.get("source_type"), "text": c.get("text")}
                for c in all_chunks
            ]
            X = model.encode(all_texts, show_progress_bar=True, convert_to_numpy=True).astype("float32")
            metas = all_metas
            texts = all_texts

        vs = VectorStore(dim, vs_dir)
        vs.add(X, metas)
    else:
        # Incremental: append to existing
        vs.add(X, metas)

    vs.save()

    # Update embed_info.json
    total_chunks = len(vs.metadatas)
    info = {
        "method": "sbert",
        "model_name": model_name,
        "dim": dim,
        "num_chunks": total_chunks,
    }
    with open(os.path.join(vs_dir, "embed_info.json"), "w") as f:
        json.dump(info, f, indent=2)

    print(f"Added {len(texts)} chunks. Total in store: {total_chunks}")
    return len(texts)


def add_text_snippet(
    text: str,
    name: str = "user_snippet",
    vs_dir: str = VECTOR_STORE_DIR,
    model_name: str = EMBED_MODEL,
):
    """Add a single text snippet to the RAG knowledge base."""
    from embeddings.vector_store import VectorStore

    chunks = _chunk_text(text, doc_name=name)
    if not chunks:
        return 0

    texts = [c["text"] for c in chunks]
    metas = chunks

    model = _get_sbert(model_name)
    X = model.encode(texts, convert_to_numpy=True).astype("float32")
    dim = X.shape[1]

    vs, info = _load_existing_store(vs_dir)
    if vs is None:
        vs = VectorStore(dim, vs_dir)

    vs.add(X, metas)
    vs.save()

    total = len(vs.metadatas)
    info_out = {
        "method": "sbert",
        "model_name": model_name,
        "dim": dim,
        "num_chunks": total,
    }
    with open(os.path.join(vs_dir, "embed_info.json"), "w") as f:
        json.dump(info_out, f, indent=2)

    print(f"Added {len(texts)} chunk(s) from '{name}'. Total: {total}")
    return len(texts)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Incrementally add docs to RAG knowledge base.")
    parser.add_argument("--add-dir", help="Directory with new .txt/.pdf files")
    parser.add_argument("--add-text", help="Raw text to add as a snippet")
    parser.add_argument("--name", default="snippet", help="Name for the text snippet")
    parser.add_argument("--force", action="store_true", help="Force full rebuild")
    parser.add_argument("--vs-dir", default=VECTOR_STORE_DIR)
    parser.add_argument("--model", default=EMBED_MODEL)
    args = parser.parse_args()

    if args.add_text:
        add_text_snippet(args.add_text, name=args.name, vs_dir=args.vs_dir, model_name=args.model)
    elif args.add_dir:
        add_documents(docs_dir=args.add_dir, vs_dir=args.vs_dir, model_name=args.model, force=args.force)
    else:
        # Default: scan the standard pdf directory for new docs
        add_documents(vs_dir=args.vs_dir, model_name=args.model, force=args.force)
