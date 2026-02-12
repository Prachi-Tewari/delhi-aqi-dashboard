"""Vector store — FAISS index + BM25 keyword index for hybrid retrieval.

Stores:
  • index.faiss   — dense vector index (FAISS or sklearn fallback)
  • metadatas.json — rich metadata per chunk (text, topic, year, credibility…)
  • bm25.pkl       — BM25 keyword index for sparse retrieval
"""

import json
import math
import pickle
import re
import numpy as np
from pathlib import Path

try:
    import faiss
    _FAISS_AVAILABLE = True
except Exception:
    _FAISS_AVAILABLE = False

from sklearn.neighbors import NearestNeighbors


# ── lightweight BM25 implementation ──────────────────────────────────

class BM25:
    """Okapi BM25 scorer — no external dependency required."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.doc_freqs: dict[str, int] = {}
        self.doc_lens: list[int] = []
        self.avg_dl: float = 0
        self.n_docs: int = 0
        self._corpus_tokens: list[list[str]] = []

    def fit(self, texts: list[str]):
        self._corpus_tokens = [self._tokenize(t) for t in texts]
        self.n_docs = len(self._corpus_tokens)
        self.doc_lens = [len(t) for t in self._corpus_tokens]
        self.avg_dl = sum(self.doc_lens) / max(self.n_docs, 1)
        self.doc_freqs = {}
        for tokens in self._corpus_tokens:
            seen = set()
            for tok in tokens:
                if tok not in seen:
                    self.doc_freqs[tok] = self.doc_freqs.get(tok, 0) + 1
                    seen.add(tok)

    def score(self, query: str) -> np.ndarray:
        q_tokens = self._tokenize(query)
        scores = np.zeros(self.n_docs, dtype="float32")
        for qt in q_tokens:
            df = self.doc_freqs.get(qt, 0)
            if df == 0:
                continue
            idf = math.log((self.n_docs - df + 0.5) / (df + 0.5) + 1)
            for i, doc_tokens in enumerate(self._corpus_tokens):
                tf = doc_tokens.count(qt)
                dl = self.doc_lens[i]
                num = tf * (self.k1 + 1)
                den = tf + self.k1 * (1 - self.b + self.b * dl / self.avg_dl)
                scores[i] += idf * num / den
        return scores

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return re.findall(r'\w+', text.lower())


# ── vector store ─────────────────────────────────────────────────────

class VectorStore:
    """FAISS dense vectors + BM25 sparse index — supports hybrid search."""

    def __init__(self, dim: int, path: str):
        self.dim = dim
        self.path = Path(path)
        self.metadatas: list[dict] = []
        self.bm25 = BM25()

        if _FAISS_AVAILABLE:
            self.index = faiss.IndexFlatL2(dim)
            self._backend = "faiss"
        else:
            self._backend = "sklearn"
            self._nn = None
            self._embs = np.zeros((0, dim), dtype="float32")

    def add(self, embeddings: np.ndarray, metadatas: list):
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)
        embeddings = embeddings.astype("float32")
        if self._backend == "faiss":
            self.index.add(embeddings)
        else:
            if self._embs.size == 0:
                self._embs = embeddings
            else:
                self._embs = np.vstack([self._embs, embeddings])
            self._nn = None
        self.metadatas.extend(metadatas)
        # Rebuild BM25 index
        self.bm25.fit([m.get("text", "") for m in self.metadatas])

    def save(self):
        self.path.mkdir(parents=True, exist_ok=True)
        if self._backend == "faiss":
            faiss.write_index(self.index, str(self.path / "index.faiss"))
        else:
            np.save(self.path / "embeddings.npy", self._embs)
        with open(self.path / "metadatas.json", "w", encoding="utf-8") as f:
            json.dump(self.metadatas, f, ensure_ascii=False, indent=2)
        with open(self.path / "bm25.pkl", "wb") as f:
            pickle.dump(self.bm25, f)

    @classmethod
    def load(cls, dim: int, path: str):
        pathp = Path(path)
        vs = cls(dim, path)
        if _FAISS_AVAILABLE and (pathp / "index.faiss").exists():
            vs.index = faiss.read_index(str(pathp / "index.faiss"))
            vs._backend = "faiss"
        else:
            emb_file = pathp / "embeddings.npy"
            if emb_file.exists():
                vs._embs = np.load(emb_file)
                vs._nn = None
                vs._backend = "sklearn"
        with open(pathp / "metadatas.json", "r", encoding="utf-8") as f:
            vs.metadatas = json.load(f)
        # Load or rebuild BM25
        bm25_path = pathp / "bm25.pkl"
        if bm25_path.exists():
            with open(bm25_path, "rb") as f:
                vs.bm25 = pickle.load(f)
        else:
            vs.bm25.fit([m.get("text", "") for m in vs.metadatas])
        return vs

    # ── search methods ──────────────────────────────────────────────

    def search(self, query_embedding, top_k: int = 5) -> list[dict]:
        """Dense (FAISS / sklearn) nearest-neighbour search."""
        q = np.array(query_embedding, dtype="float32").reshape(1, -1)
        results = []
        if self._backend == "faiss":
            D, I = self.index.search(q, min(top_k, len(self.metadatas)))
            for rank, idx in enumerate(I[0]):
                if 0 <= idx < len(self.metadatas):
                    r = dict(self.metadatas[idx])
                    r["dense_score"] = float(1.0 / (1.0 + D[0][rank]))
                    results.append(r)
        else:
            if self._embs.size == 0:
                return []
            k = min(top_k, len(self._embs))
            if self._nn is None:
                self._nn = NearestNeighbors(n_neighbors=k, metric="euclidean")
                self._nn.fit(self._embs)
            dist, idxs = self._nn.kneighbors(q, n_neighbors=k)
            for rank, idx in enumerate(idxs[0]):
                if 0 <= idx < len(self.metadatas):
                    r = dict(self.metadatas[idx])
                    r["dense_score"] = float(1.0 / (1.0 + dist[0][rank]))
                    results.append(r)
        return results

    def search_bm25(self, query: str, top_k: int = 5) -> list[dict]:
        """Sparse (BM25) keyword search."""
        scores = self.bm25.score(query)
        if scores.max() == 0:
            return []
        # Normalize to [0, 1]
        norm = scores / (scores.max() + 1e-9)
        top_idxs = np.argsort(-norm)[:top_k]
        results = []
        for idx in top_idxs:
            if norm[idx] > 0 and idx < len(self.metadatas):
                r = dict(self.metadatas[idx])
                r["bm25_score"] = float(norm[idx])
                results.append(r)
        return results

    def hybrid_search(
        self,
        query_embedding,
        query_text: str,
        top_k: int = 20,
        dense_weight: float = 0.6,
        sparse_weight: float = 0.4,
    ) -> list[dict]:
        """Reciprocal Rank Fusion of dense + sparse search."""
        dense_results = self.search(query_embedding, top_k=top_k)
        sparse_results = self.search_bm25(query_text, top_k=top_k)

        # RRF scoring
        rrf_k = 60
        scores: dict[int, float] = {}
        meta_map: dict[int, dict] = {}

        for rank, r in enumerate(dense_results):
            text_hash = hash(r.get("text", ""))
            rrf = dense_weight / (rrf_k + rank + 1)
            scores[text_hash] = scores.get(text_hash, 0) + rrf
            if text_hash not in meta_map:
                meta_map[text_hash] = r

        for rank, r in enumerate(sparse_results):
            text_hash = hash(r.get("text", ""))
            rrf = sparse_weight / (rrf_k + rank + 1)
            scores[text_hash] = scores.get(text_hash, 0) + rrf
            if text_hash not in meta_map:
                meta_map[text_hash] = r

        # Sort by combined RRF score
        sorted_ids = sorted(scores, key=lambda h: -scores[h])
        results = []
        for h in sorted_ids[:top_k]:
            r = dict(meta_map[h])
            r["hybrid_score"] = round(scores[h], 6)
            results.append(r)
        return results
