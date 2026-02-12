"""Embedding cache layer — avoids re-encoding repeated or similar queries.

Uses an LRU dict keyed by text hash.  Persists to disk as a .npz file
so the cache survives restarts.
"""

import hashlib
import json
import numpy as np
from pathlib import Path
from collections import OrderedDict
from threading import Lock


class EmbeddingCache:
    """Thread-safe LRU embedding cache with optional disk persistence."""

    def __init__(self, max_size: int = 2048, cache_path: str | None = None):
        self._max = max_size
        self._cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self._lock = Lock()
        self._path = Path(cache_path) if cache_path else None
        self._hits = 0
        self._misses = 0
        if self._path and self._path.exists():
            self._load_from_disk()

    # ── public API ──────────────────────────────────────────────────

    def get(self, text: str) -> np.ndarray | None:
        key = self._key(text)
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                self._hits += 1
                return self._cache[key].copy()
            self._misses += 1
            return None

    def put(self, text: str, embedding: np.ndarray):
        key = self._key(text)
        with self._lock:
            self._cache[key] = embedding.astype("float32")
            self._cache.move_to_end(key)
            if len(self._cache) > self._max:
                self._cache.popitem(last=False)

    def get_many(self, texts: list[str]) -> tuple[np.ndarray | None, list[int]]:
        """Return cached embeddings where available.

        Returns:
            (partial_matrix, missing_indices): partial_matrix has cached rows
            filled in and zeros for missing ones; missing_indices lists the
            positions that still need embedding.
        """
        dim = None
        results = {}
        missing = []
        for i, t in enumerate(texts):
            emb = self.get(t)
            if emb is not None:
                results[i] = emb
                if dim is None:
                    dim = emb.shape[0]
            else:
                missing.append(i)
        if dim is None:
            return None, list(range(len(texts)))
        mat = np.zeros((len(texts), dim), dtype="float32")
        for idx, emb in results.items():
            mat[idx] = emb
        return mat, missing

    def put_many(self, texts: list[str], embeddings: np.ndarray):
        for t, emb in zip(texts, embeddings):
            self.put(t, emb)

    def save(self):
        if self._path is None:
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        keys = list(self._cache.keys())
        vals = np.stack(list(self._cache.values())) if self._cache else np.array([])
        np.savez_compressed(
            self._path,
            keys=np.array(keys, dtype=object),
            vals=vals,
        )

    @property
    def stats(self) -> dict:
        total = self._hits + self._misses
        return {
            "size": len(self._cache),
            "max_size": self._max,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": round(self._hits / total, 3) if total else 0.0,
        }

    # ── internals ───────────────────────────────────────────────────

    @staticmethod
    def _key(text: str) -> str:
        return hashlib.sha256(text.strip().lower().encode()).hexdigest()[:24]

    def _load_from_disk(self):
        try:
            data = np.load(self._path, allow_pickle=True)
            keys = data["keys"]
            vals = data["vals"]
            if vals.ndim == 2:
                for k, v in zip(keys, vals):
                    self._cache[str(k)] = v.astype("float32")
        except Exception:
            pass
