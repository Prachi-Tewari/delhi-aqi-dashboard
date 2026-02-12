"""Cross-encoder reranker with adaptive top-k and source reliability weighting.

Pipeline:
  1. Receive top-N candidates from hybrid retriever
  2. Score each with cross-encoder/ms-marco-MiniLM-L-6-v2
  3. Apply source reliability weight (from metadata credibility_score)
  4. Adaptive top-k: return only results above a confidence floor
"""

from __future__ import annotations

import math
import numpy as np


_CROSS_ENCODER = None
_CE_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def _load_cross_encoder():
    global _CROSS_ENCODER
    if _CROSS_ENCODER is None:
        try:
            from sentence_transformers import CrossEncoder
            _CROSS_ENCODER = CrossEncoder(_CE_MODEL_NAME)
        except Exception as e:
            print(f"[reranker] Could not load cross-encoder: {e}")
    return _CROSS_ENCODER


def rerank(
    query: str,
    candidates: list[dict],
    top_k: int = 5,
    min_score: float = 0.15,
    reliability_weight: float = 0.15,
) -> list[dict]:
    """Rerank candidates using cross-encoder + source reliability.

    Args:
        query: The user's question.
        candidates: List of dicts with at least a 'text' field.
        top_k: Maximum results to return.
        min_score: Minimum combined score to include (adaptive cutoff).
        reliability_weight: How much to factor in credibility_score (0-1).

    Returns:
        Reranked list of candidates with added 'rerank_score',
        'cross_encoder_score', and 'final_score' fields.
    """
    if not candidates:
        return []

    ce = _load_cross_encoder()
    if ce is None:
        # Fallback: just return candidates sorted by credibility
        for c in candidates:
            c["rerank_score"] = c.get("credibility_score", 0.75)
            c["cross_encoder_score"] = 0.0
            c["final_score"] = c["rerank_score"]
        return sorted(candidates, key=lambda x: -x["final_score"])[:top_k]

    # Prepare cross-encoder pairs
    pairs = [(query, c.get("text", "")) for c in candidates]
    try:
        ce_scores = ce.predict(pairs)
    except Exception:
        ce_scores = [0.0] * len(candidates)

    # Normalize CE scores to [0, 1] using sigmoid
    ce_scores_norm = [_sigmoid(s) for s in ce_scores]

    # Combine with source reliability
    for i, c in enumerate(candidates):
        ce_norm = ce_scores_norm[i]
        credibility = c.get("credibility_score", 0.75)

        # Weighted combination
        combined = (1 - reliability_weight) * ce_norm + reliability_weight * credibility

        c["cross_encoder_score"] = round(float(ce_scores[i]), 4)
        c["rerank_score"] = round(ce_norm, 4)
        c["final_score"] = round(combined, 4)

    # Sort by final score descending
    ranked = sorted(candidates, key=lambda x: -x["final_score"])

    # Adaptive top-k: return at most top_k, but cut off below min_score
    result = []
    for c in ranked:
        if len(result) >= top_k:
            break
        if c["final_score"] >= min_score or len(result) == 0:
            result.append(c)

    return result


def adaptive_top_k(
    scores: list[float],
    max_k: int = 10,
    min_k: int = 2,
    score_gap_threshold: float = 0.3,
) -> int:
    """Determine optimal k based on score distribution.

    Looks for a significant drop in scores (elbow detection).
    """
    if len(scores) <= min_k:
        return len(scores)

    sorted_scores = sorted(scores, reverse=True)

    for i in range(1, min(len(sorted_scores), max_k)):
        gap = sorted_scores[i - 1] - sorted_scores[i]
        relative_gap = gap / (sorted_scores[0] + 1e-9)
        if relative_gap > score_gap_threshold and i >= min_k:
            return i

    return min(max_k, len(scores))


def _sigmoid(x: float) -> float:
    """Numerically stable sigmoid."""
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    else:
        ex = math.exp(x)
        return ex / (1.0 + ex)


if __name__ == "__main__":
    # Quick test
    test_candidates = [
        {"text": "PM2.5 causes respiratory diseases in Delhi", "credibility_score": 0.9},
        {"text": "The weather in Delhi is hot in summer", "credibility_score": 0.7},
        {"text": "Delhi AQI reached 500 during Diwali 2023", "credibility_score": 0.85},
    ]
    results = rerank("health effects of PM2.5", test_candidates)
    for r in results:
        print(f"  score={r['final_score']:.3f}  ce={r['cross_encoder_score']:.3f}  {r['text'][:60]}")
