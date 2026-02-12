"""Response confidence scoring.

Computes a 0–1 confidence score for each LLM response based on:
  • Reranker scores of retrieved sources
  • Number of supporting references
  • Source credibility scores
  • Query-response relevance heuristics
  • Citation density
"""

from __future__ import annotations

import re
import math


def compute_confidence(
    response: str,
    retrieved_chunks: list[dict],
    query: str,
    intent: str = "general",
) -> dict:
    """Compute a composite confidence score for the response.

    Returns:
        {
            "score": float (0-1),
            "grade": str ("high", "medium", "low"),
            "breakdown": {
                "source_quality": float,
                "citation_coverage": float,
                "relevance": float,
                "response_quality": float,
            },
            "explanation": str,
        }
    """
    # 1. Source quality — average reranker + credibility scores
    source_quality = _source_quality_score(retrieved_chunks)

    # 2. Citation coverage — how well the response cites sources
    citation_coverage = _citation_score(response, retrieved_chunks)

    # 3. Relevance — does the response address the query?
    relevance = _relevance_score(response, query)

    # 4. Response quality — length, structure, specificity
    response_quality = _response_quality_score(response)

    # Weighted combination (adjust weights by intent)
    weights = _get_weights(intent)
    composite = (
        weights["source"] * source_quality
        + weights["citation"] * citation_coverage
        + weights["relevance"] * relevance
        + weights["quality"] * response_quality
    )
    composite = min(max(composite, 0.0), 1.0)

    grade = "high" if composite >= 0.7 else ("medium" if composite >= 0.4 else "low")

    explanations = []
    if source_quality > 0.7:
        explanations.append("Strong source material available")
    elif source_quality < 0.3:
        explanations.append("Limited source material")

    if citation_coverage > 0.6:
        explanations.append("Well-cited response")
    elif citation_coverage < 0.2 and retrieved_chunks:
        explanations.append("Response lacks source citations")

    if relevance > 0.6:
        explanations.append("Directly addresses the query")
    elif relevance < 0.3:
        explanations.append("Response may not fully address the query")

    return {
        "score": round(composite, 3),
        "grade": grade,
        "breakdown": {
            "source_quality": round(source_quality, 3),
            "citation_coverage": round(citation_coverage, 3),
            "relevance": round(relevance, 3),
            "response_quality": round(response_quality, 3),
        },
        "explanation": "; ".join(explanations) if explanations else "Standard confidence",
    }


def format_confidence_badge(confidence: dict) -> str:
    """Return a markdown badge string for the confidence score."""
    score = confidence["score"]
    grade = confidence["grade"]
    emoji = {"high": "[HIGH]", "medium": "[MED]", "low": "[LOW]"}[grade]
    pct = int(score * 100)
    return f"{emoji} Confidence: {pct}% ({grade})"


# ── internal scoring helpers ────────────────────────────────────────

def _source_quality_score(chunks: list[dict]) -> float:
    if not chunks:
        return 0.2  # base score when no sources available

    scores = []
    for c in chunks:
        # Use reranker final_score if available, else credibility
        final = c.get("final_score", c.get("rerank_score", 0))
        cred = c.get("credibility_score", 0.75)
        scores.append(0.7 * final + 0.3 * cred)

    # Weight top sources more heavily
    scores.sort(reverse=True)
    weighted = sum(s * (0.9 ** i) for i, s in enumerate(scores))
    normalizer = sum(0.9 ** i for i in range(len(scores)))
    return min(weighted / (normalizer + 1e-9), 1.0)


def _citation_score(response: str, chunks: list[dict]) -> float:
    if not chunks:
        return 0.5  # neutral when no sources to cite

    ref_citations = len(re.findall(r'\[Ref\s*\d+\]', response))
    url_citations = len(re.findall(r'\[.*?\]\(https?://.*?\)', response))
    total_citations = ref_citations + url_citations

    n_sources = len(chunks)
    coverage = min(total_citations / max(n_sources, 1), 1.0)

    # Bonus for having some citations
    if total_citations > 0:
        coverage = max(coverage, 0.4)

    return coverage


def _relevance_score(response: str, query: str) -> float:
    """Simple keyword overlap heuristic for relevance."""
    if not response or not query:
        return 0.3

    q_words = set(query.lower().split())
    r_words = set(response.lower().split())

    # Remove stopwords (rough)
    stop = {"the", "a", "an", "is", "are", "was", "were", "in", "on", "at",
            "to", "for", "of", "and", "or", "it", "this", "that", "with",
            "what", "how", "why", "when", "where", "which", "do", "does",
            "can", "could", "would", "should", "about", "from", "by", "as"}
    q_words -= stop
    r_words -= stop

    if not q_words:
        return 0.5

    overlap = len(q_words & r_words)
    score = overlap / len(q_words)

    # Boost if response is substantial
    if len(response) > 200:
        score = min(score + 0.1, 1.0)

    return min(score, 1.0)


def _response_quality_score(response: str) -> float:
    """Judge response structural quality."""
    if not response:
        return 0.0

    score = 0.3  # base

    # Length — not too short, not too long
    words = len(response.split())
    if 50 < words < 800:
        score += 0.2
    elif words >= 800:
        score += 0.15
    elif words > 20:
        score += 0.1

    # Has markdown formatting
    if re.search(r'^\s*[-*•]', response, re.MULTILINE):
        score += 0.1
    if re.search(r'^#+\s', response, re.MULTILINE):
        score += 0.1

    # Has specific data / numbers
    if re.findall(r'\b\d+\.?\d*\b', response):
        score += 0.1

    # Has links
    if re.findall(r'\[.*?\]\(https?://.*?\)', response):
        score += 0.1

    return min(score, 1.0)


def _get_weights(intent: str) -> dict:
    """Intent-specific weight distribution."""
    configs = {
        "live_data":     {"source": 0.2, "citation": 0.1, "relevance": 0.4, "quality": 0.3},
        "health_advice": {"source": 0.3, "citation": 0.3, "relevance": 0.2, "quality": 0.2},
        "historical":    {"source": 0.35, "citation": 0.25, "relevance": 0.2, "quality": 0.2},
        "comparison":    {"source": 0.3, "citation": 0.25, "relevance": 0.25, "quality": 0.2},
        "factual":       {"source": 0.3, "citation": 0.25, "relevance": 0.25, "quality": 0.2},
        "policy":        {"source": 0.3, "citation": 0.3, "relevance": 0.2, "quality": 0.2},
        "general":       {"source": 0.15, "citation": 0.1, "relevance": 0.35, "quality": 0.4},
    }
    return configs.get(intent, configs["general"])
