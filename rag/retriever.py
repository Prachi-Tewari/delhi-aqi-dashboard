"""Advanced retriever — hybrid search (FAISS + BM25) → cross-encoder reranking.

Pipeline:
  1. Embed query with BAAI/bge-base-en-v1.5 (with cache)
  2. Hybrid search: FAISS dense + BM25 sparse → top-20 candidates
  3. Cross-encoder rerank (ms-marco-MiniLM-L-6-v2) + source reliability weighting
  4. Adaptive top-k: return only confidently relevant results
  5. Optional topic boost from query classifier
"""

import os
import json
import pickle
import numpy as np

from embeddings.vector_store import VectorStore
from embeddings.cache import EmbeddingCache

# ── model constants ──────────────────────────────────────────────────

BGE_MODEL = "BAAI/bge-base-en-v1.5"
BGE_DIM = 768
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

_EMBED_CACHE: EmbeddingCache | None = None


def _get_cache(path: str = "embeddings/vector_store/embed_cache.npz") -> EmbeddingCache:
    global _EMBED_CACHE
    if _EMBED_CACHE is None:
        _EMBED_CACHE = EmbeddingCache(max_size=4096, cache_path=path)
    return _EMBED_CACHE


def _load_sbert(model_name: str):
    try:
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer(model_name)
    except Exception:
        return None


class Retriever:
    """Hybrid retriever with caching, reranking, and adaptive top-k."""

    def __init__(
        self,
        vectorstore_path: str = "embeddings/vector_store",
        embed_model: str = BGE_MODEL,
    ):
        self.vectorstore_path = vectorstore_path
        self._sbert = None

        # Load embed_info.json
        info_path = os.path.join(vectorstore_path, "embed_info.json")
        self._method = "sbert"
        self._dim = BGE_DIM

        if os.path.exists(info_path):
            with open(info_path) as f:
                info = json.load(f)
            self._method = info.get("method", "sbert")
            self._dim = info.get("dim", BGE_DIM)
            embed_model = info.get("model_name", embed_model)

        if self._method == "sbert":
            self._sbert = _load_sbert(embed_model)
            if self._sbert is not None and self._dim is None:
                self._dim = self._sbert.get_sentence_embedding_dimension()

        if self._dim is None:
            self._dim = BGE_DIM

        # Load vector store
        try:
            self.vs = VectorStore.load(self._dim, vectorstore_path)
        except Exception:
            self.vs = None

        self._cache = _get_cache(
            os.path.join(vectorstore_path, "embed_cache.npz")
        )

    # ── embedding ───────────────────────────────────────────────────

    def _embed_query(self, query: str) -> np.ndarray:
        """Embed query with BGE prefix and cache."""
        # BGE models need a query prefix for retrieval
        full_query = BGE_QUERY_PREFIX + query

        # Check cache
        cached = self._cache.get(full_query)
        if cached is not None:
            return cached

        if self._sbert is not None:
            emb = self._sbert.encode([full_query])[0].astype("float32")
        else:
            raise RuntimeError("No embedding model available.")

        self._cache.put(full_query, emb)
        return emb

    # ── retrieval pipeline ──────────────────────────────────────────

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        use_reranker: bool = True,
        intent_config: dict | None = None,
    ) -> list[dict]:
        """Full retrieval pipeline: hybrid search → rerank → adaptive top-k.

        Args:
            query: User question.
            top_k: Target number of results.
            use_reranker: Whether to apply cross-encoder reranking.
            intent_config: From query_classifier, adjusts behavior.

        Returns:
            List of result dicts with text, scores, metadata.
        """
        if self.vs is None:
            return []

        # Phase 1: Hybrid search — retrieve 20 candidates
        candidate_k = max(top_k * 4, 20)
        q_emb = self._embed_query(query)
        candidates = self.vs.hybrid_search(
            q_emb, query, top_k=candidate_k,
            dense_weight=0.6, sparse_weight=0.4,
        )

        if not candidates:
            return []

        # Optional: boost by preferred topics from intent
        if intent_config and "preferred_topics" in intent_config:
            preferred = set(intent_config["preferred_topics"])
            for c in candidates:
                if c.get("topic") in preferred:
                    c["hybrid_score"] = c.get("hybrid_score", 0) * 1.3

        # Phase 2: Rerank with cross-encoder
        if use_reranker and len(candidates) > 1:
            try:
                from rag.reranker import rerank
                results = rerank(
                    query, candidates,
                    top_k=top_k,
                    min_score=0.15,
                    reliability_weight=0.15,
                )
            except Exception:
                # Fallback: just return top-k by hybrid score
                results = sorted(candidates, key=lambda x: -x.get("hybrid_score", 0))[:top_k]
        else:
            results = sorted(candidates, key=lambda x: -x.get("hybrid_score", 0))[:top_k]

        return results

    def retrieve_simple(self, query: str, top_k: int = 5) -> list[dict]:
        """Simple dense-only retrieval (backwards compatible)."""
        if self.vs is None:
            return []
        q_emb = self._embed_query(query)
        return self.vs.search(q_emb, top_k=top_k)

    def save_cache(self):
        """Persist the embedding cache to disk."""
        if self._cache:
            self._cache.save()


if __name__ == "__main__":
    r = Retriever("embeddings/vector_store")
    queries = [
        "health effects of pm2.5 in delhi",
        "what is the current AQI?",
        "compare Delhi and Beijing pollution",
        "government policies for stubble burning",
    ]
    for q in queries:
        results = r.retrieve(q, top_k=3)
        print(f"\nQ: {q}")
        for doc in results:
            score = doc.get("final_score", doc.get("hybrid_score", "?"))
            print(f"  [{score:.3f}] {doc.get('doc_name')} — {doc.get('topic', '?')}")
