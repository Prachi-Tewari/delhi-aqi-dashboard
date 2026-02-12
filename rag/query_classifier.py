"""Query intent classifier — detects what kind of answer the user needs.

Intent types:
  • live_data      — current AQI, pollutant levels, real-time readings
  • health_advice  — safety, exercise, masks, vulnerable groups
  • historical     — trends, past data, year comparisons
  • comparison     — city vs city, pollutant vs WHO, methodology diff
  • factual        — definitions, how AQI works, what is PM2.5
  • policy         — government actions, GRAP, regulations
  • general        — casual / off-topic conversation

The classifier uses keyword scoring + simple heuristics (no ML model needed).
"""

from __future__ import annotations

import re


INTENT_KEYWORDS: dict[str, list[str]] = {
    "live_data": [
        r"current|right now|today|live|real.?time|latest|now|at the moment",
        r"what is.*(aqi|pm|pollution)|how.*(air|pollution).*(today|now)",
    ],
    "health_advice": [
        r"safe|health|jog|run|exercise|outdoor|mask|purifier|child|pregnan|asthma|copd",
        r"should i|can i|is it ok|advic|recommend|protect|risk",
    ],
    "historical": [
        r"history|historical|trend|past|year|month|season|winter|summer|over time",
        r"201\d|202\d|decade|annual|data from|last year|compare.*year",
    ],
    "comparison": [
        r"compar|versus|vs\.?|differ|better|worse|rank|city|beijing|mumbai|who",
        r"standard|guideline|epa|naqi|limit",
    ],
    "factual": [
        r"what is|define|explain|how does|meaning|calcula|formula|breakpoint",
        r"why|cause|source|factor|reason|science|method",
    ],
    "policy": [
        r"polic|government|regulation|grap|cpcb|ngt|ban|odd.?even|stubble",
        r"initiative|scheme|law|court|tribunal|ministry",
    ],
}

# Map intents to retrieval strategies
INTENT_CONFIG: dict[str, dict] = {
    "live_data": {
        "needs_api": True,
        "needs_rag": False,
        "top_k": 3,
        "description": "Real-time AQI data query",
    },
    "health_advice": {
        "needs_api": True,   # combine live data with health advice
        "needs_rag": True,
        "top_k": 5,
        "preferred_topics": ["health", "exercise_outdoor"],
        "description": "Health and safety advice",
    },
    "historical": {
        "needs_api": False,
        "needs_rag": True,
        "top_k": 7,
        "preferred_topics": ["historical_data"],
        "description": "Historical trend analysis",
    },
    "comparison": {
        "needs_api": True,
        "needs_rag": True,
        "top_k": 6,
        "preferred_topics": ["aqi_methodology"],
        "description": "Comparative analysis",
    },
    "factual": {
        "needs_api": False,
        "needs_rag": True,
        "top_k": 5,
        "description": "Factual / definitional query",
    },
    "policy": {
        "needs_api": False,
        "needs_rag": True,
        "top_k": 5,
        "preferred_topics": ["policy"],
        "description": "Policy and regulation query",
    },
    "general": {
        "needs_api": False,
        "needs_rag": False,
        "top_k": 3,
        "description": "General conversation",
    },
}


def classify_query(query: str) -> dict:
    """Classify query intent and return config for retrieval pipeline.

    Returns:
        {
            "intent": str,
            "confidence": float,
            "config": dict,     # from INTENT_CONFIG
            "all_scores": dict, # raw scores for transparency
        }
    """
    q_lower = query.lower().strip()

    scores: dict[str, float] = {}
    for intent, patterns in INTENT_KEYWORDS.items():
        score = 0.0
        for pat in patterns:
            matches = re.findall(pat, q_lower)
            score += len(matches) * 1.5
        scores[intent] = score

    total = sum(scores.values())
    if total == 0:
        return {
            "intent": "general",
            "confidence": 0.3,
            "config": INTENT_CONFIG["general"],
            "all_scores": scores,
        }

    best_intent = max(scores, key=scores.get)
    confidence = min(scores[best_intent] / (total + 1e-9), 1.0)

    # If confidence is very low, default to general
    if confidence < 0.2 or scores[best_intent] < 1.0:
        best_intent = "general"
        confidence = 0.3

    return {
        "intent": best_intent,
        "confidence": round(confidence, 3),
        "config": INTENT_CONFIG.get(best_intent, INTENT_CONFIG["general"]),
        "all_scores": {k: round(v, 2) for k, v in scores.items()},
    }


def needs_live_api(query: str) -> bool:
    """Quick check: does this query need live AQI data?"""
    return classify_query(query)["config"].get("needs_api", False)


def needs_rag(query: str) -> bool:
    """Quick check: does this query need RAG retrieval?"""
    return classify_query(query)["config"].get("needs_rag", True)


if __name__ == "__main__":
    test_queries = [
        "What is the current AQI in Delhi?",
        "Is it safe to jog outside today?",
        "How has Delhi's air quality changed over the last decade?",
        "Compare Delhi AQI with WHO guidelines",
        "What is PM2.5 and why does it matter?",
        "What is the government doing about stubble burning?",
        "Tell me a joke",
    ]
    for q in test_queries:
        r = classify_query(q)
        print(f"\nQ: {q}")
        print(f"  Intent: {r['intent']} (conf={r['confidence']})")
        print(f"  Config: {r['config']}")
