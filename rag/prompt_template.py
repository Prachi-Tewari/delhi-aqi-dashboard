"""Build the user-message prompt for Groq / LLM chat API.

The system message lives in llm_pipeline.py; this module constructs
the *user* message that combines:
  - User's question (front-and-centre)
  - Query intent classification
  - Live API data (from tool calls)
  - RAG references with confidence scores
  - Lightweight guidance
"""


def build_prompt(
    question: str,
    retrieved_chunks: list,
    live_snapshot: dict,
    intent: dict | None = None,
    tool_result: dict | None = None,
    confidence_hint: str | None = None,
) -> str:
    """Assemble all context into one prompt.

    The prompt puts the user's actual question front-and-centre and only
    provides data/references as *optional context* the LLM can draw on.
    """

    parts = []

    # ── The actual question comes FIRST ─────────────────────────────
    parts.append(f"**My question:** {question}")

    # ── Intent hint (helps LLM calibrate response style) ────────────
    if intent:
        intent_type = intent.get("intent", "general")
        conf = intent.get("confidence", 0)
        parts.append(f"\n*[Query intent: {intent_type} (conf={conf:.0%})]*")

    # ── Tool-call results (live API data) ───────────────────────────
    if tool_result and tool_result.get("success"):
        from rag.tool_calling import format_tool_result
        formatted = format_tool_result(tool_result)
        if formatted:
            parts.append(f"\n---\n{formatted}")

    # ── Supporting context (live data snapshot) ─────────────────────
    elif live_snapshot:
        parts.append(
            "\n---\n**Context — Live AQI readings for Delhi right now** "
            "(use only if relevant to my question):"
        )
        for k, v in sorted(live_snapshot.items()):
            parts.append(f"  • {k}: {v}")

    # ── Supporting context (RAG knowledge base) ─────────────────────
    if retrieved_chunks:
        parts.append(
            "\n**Context — Knowledge-base excerpts** "
            "(cite as [Ref N] only when you actually use them):"
        )
        for i, r in enumerate(retrieved_chunks, 1):
            snippet = r.get("text", "")[:1200]
            src = r.get("doc_name", "unknown")
            topic = r.get("topic", "")
            score = r.get("final_score", r.get("rerank_score", ""))
            score_str = f" | relevance={score:.2f}" if isinstance(score, float) else ""
            topic_str = f" | topic={topic}" if topic else ""
            parts.append(f"\n[Ref {i}] ({src}{topic_str}{score_str}):\n{snippet}")

    # ── Lightweight guidance ────────────────────────────────────────
    parts.append(
        "\n---\n**How to answer:**\n"
        "• Answer **exactly what I asked** — do NOT produce a generic AQI "
        "report unless I specifically asked for one.\n"
        "• Use the live data and references above only when they are "
        "relevant to my question; otherwise rely on your own knowledge.\n"
        "• If you reference external facts, include **clickable source links** "
        "in markdown: [text](https://url).\n"
        "• Use [Ref N] citations when referencing knowledge-base excerpts.\n"
        "• Use markdown formatting for readability.\n"
        "• Be detailed and helpful but stay on-topic."
    )

    return "\n".join(parts)
