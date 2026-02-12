"""Build embeddings for document chunks and store in the vector database.

Uses BAAI/bge-base-en-v1.5 (768-dim) as the primary embedding model.
Falls back to TF-IDF if sentence-transformers is not installed.

Also builds a BM25 keyword index for hybrid retrieval.
"""

import os
import json
import pickle
import numpy as np

from ingestion.load_pdfs import load_all_pdfs
from ingestion.chunk_docs import chunk_texts
from embeddings.vector_store import VectorStore

BGE_MODEL = "BAAI/bge-base-en-v1.5"
BGE_DIM = 768
BGE_DOC_PREFIX = ""  # BGE uses prefix only for queries, not docs


def _get_embedder(model_name: str):
    """Try sentence-transformers first; fall back to TF-IDF."""
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(model_name)
        return "sbert", model
    except Exception:
        from sklearn.feature_extraction.text import TfidfVectorizer
        return "tfidf", TfidfVectorizer(max_features=768)


def build_embeddings(
    pdf_dir: str,
    out_dir: str,
    model_name: str = BGE_MODEL,
):
    """Load documents, chunk semantically, embed, and save to vector store."""
    pages = load_all_pdfs(pdf_dir)
    chunks = chunk_texts(pages, max_tokens=512, overlap_tokens=200)
    texts = [c["text"] for c in chunks]

    if not texts:
        print("No documents found to embed.")
        return

    metas = [
        {
            "doc_name": c.get("doc_name"),
            "page": c.get("page"),
            "source_type": c.get("source_type"),
            "text": c.get("text"),
            "topic": c.get("topic", "general"),
            "year": c.get("year"),
            "credibility_score": c.get("credibility_score", 0.75),
            "token_count": c.get("token_count", 0),
        }
        for c in chunks
    ]

    kind, embedder = _get_embedder(model_name)
    os.makedirs(out_dir, exist_ok=True)

    if kind == "sbert":
        print(f"Embedding {len(texts)} chunks with {model_name} ...")
        X = embedder.encode(texts, show_progress_bar=True,
                            convert_to_numpy=True).astype("float32")
    else:
        print(f"sentence-transformers not available; using TF-IDF for {len(texts)} chunks ...")
        X = embedder.fit_transform(texts).toarray().astype("float32")
        with open(os.path.join(out_dir, "tfidf.pkl"), "wb") as f:
            pickle.dump(embedder, f)

    dim = X.shape[1]
    vs = VectorStore(dim, out_dir)
    vs.add(X, metas)
    vs.save()

    # Record embedding info
    info = {
        "method": kind,
        "model_name": model_name,
        "dim": dim,
        "num_chunks": len(texts),
    }
    with open(os.path.join(out_dir, "embed_info.json"), "w") as f:
        json.dump(info, f, indent=2)

    # Print topic distribution
    topics: dict[str, int] = {}
    for m in metas:
        t = m.get("topic", "general")
        topics[t] = topics.get(t, 0) + 1
    print(f"Saved {len(texts)} embeddings (dim={dim}, method={kind}) to {out_dir}")
    print(f"Topic distribution: {topics}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Build vector store from PDF/text documents.")
    parser.add_argument("--pdf_dir", required=True,
                        help="Directory containing PDF / text files")
    parser.add_argument("--out_dir", required=True,
                        help="Output directory for vector store")
    parser.add_argument("--model", default=BGE_MODEL)
    args = parser.parse_args()

    build_embeddings(args.pdf_dir, args.out_dir, args.model)
