"""Hallucination detection — checks if the LLM response properly cites sources.

When RAG context was provided but the response contains no source citations,
this module triggers a regeneration with explicit instructions.
"""

from __future__ import annotations

import re


def detect_hallucination(
    response: str,
    retrieved_chunks: list[dict],
    query: str,
) -> dict:
    """Analyze a response for potential hallucination.

    Returns:
        {
            "has_citations": bool,
            "citation_count": int,
            "expected_citations": bool,  # True if RAG was provided
            "hallucination_risk": str,   # "low", "medium", "high"
            "should_regenerate": bool,
            "reasons": list[str],
        }
    """
    has_rag = bool(retrieved_chunks)
    citations = re.findall(r'\[Ref\s*\d+\]', response)
    citation_count = len(citations)
    has_citations = citation_count > 0

    # Check for source links
    has_links = bool(re.findall(r'\[.*?\]\(https?://.*?\)', response))

    # Check for hedging language (signs of uncertainty)
    hedging = len(re.findall(
        r"i think|probably|maybe|not sure|might be|possibly|i believe|arguably",
        response.lower()
    ))

    # Check for very specific numeric claims without sources
    numeric_claims = re.findall(r'\b\d{3,}\b', response)
    unsourced_numbers = len(numeric_claims) > 3 and not has_citations and not has_links

    reasons = []
    risk = "low"

    if has_rag and not has_citations:
        reasons.append("RAG context provided but no [Ref N] citations in response")
        risk = "medium"

    if unsourced_numbers:
        reasons.append(f"Contains {len(numeric_claims)} numeric claims without source attribution")
        risk = "high" if risk == "medium" else "medium"

    if hedging > 2:
        reasons.append(f"High hedging language ({hedging} instances)")
        risk = "medium" if risk == "low" else risk

    if has_rag and len(retrieved_chunks) >= 3 and not has_citations and not has_links:
        risk = "high"
        reasons.append("Multiple relevant sources available but none cited")

    should_regenerate = (risk == "high" and has_rag)

    return {
        "has_citations": has_citations,
        "citation_count": citation_count,
        "has_links": has_links,
        "expected_citations": has_rag,
        "hallucination_risk": risk,
        "should_regenerate": should_regenerate,
        "reasons": reasons,
    }


def build_regeneration_prompt(query: str, retrieved_chunks: list[dict]) -> str:
    """Build a stricter prompt that forces source citation."""
    rag_context = ""
    for i, r in enumerate(retrieved_chunks, 1):
        snippet = r.get("text", "")[:1000]
        src = r.get("doc_name", "unknown")
        rag_context += f"\n[Ref {i}] ({src}): {snippet}\n"

    return (
        f"IMPORTANT: Your previous answer did not cite any sources. "
        f"Please answer again with proper citations.\n\n"
        f"Question: {query}\n\n"
        f"You MUST use [Ref N] citations from the sources below. "
        f"If a fact comes from a source, cite it inline.\n\n"
        f"Sources:\n{rag_context}\n\n"
        f"Rules:\n"
        f"1. Every factual claim must have a [Ref N] citation\n"
        f"2. If you're unsure, say so rather than guessing\n"
        f"3. Use markdown formatting\n"
        f"4. Be concise but thorough"
    )
