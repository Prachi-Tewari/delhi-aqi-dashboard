"""RAGAS-style evaluation script for the AQI RAG system.

Metrics implemented:
  • Faithfulness   — Does the answer only contain info from the context?
  • Answer Relevancy — Is the answer relevant to the question?
  • Context Precision — Are retrieved docs relevant to the query?
  • Context Recall   — Does the context cover the reference answer?
  • Answer Correctness — Is the answer factually correct vs reference?

Usage:
  python -m evaluation.ragas_eval                     # run default test set
  python -m evaluation.ragas_eval --test-file my.json # custom test set
  python -m evaluation.ragas_eval --quick             # just 3 test cases
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ── Test dataset ─────────────────────────────────────────────────────

DEFAULT_TEST_SET = [
    {
        "question": "What are the health effects of PM2.5 on Delhi residents?",
        "reference_answer": "PM2.5 causes respiratory diseases, cardiovascular problems, reduced lung function, increased mortality, and aggravates asthma. Long-term exposure reduces life expectancy by several years.",
        "expected_topics": ["health"],
    },
    {
        "question": "How is AQI calculated using India's NAQI standard?",
        "reference_answer": "India NAQI uses sub-index interpolation for 8 pollutants (PM2.5, PM10, NO2, SO2, CO, O3, NH3, Pb). Each pollutant gets a sub-index using breakpoint tables, and the overall AQI is the highest sub-index.",
        "expected_topics": ["aqi_methodology"],
    },
    {
        "question": "What causes Delhi's severe winter pollution?",
        "reference_answer": "Delhi winter pollution is caused by temperature inversions trapping pollutants, stubble burning in Punjab/Haryana, vehicular emissions, construction dust, industrial pollution, and unfavorable meteorological conditions.",
        "expected_topics": ["meteorology", "pollution_sources"],
    },
    {
        "question": "Is it safe to exercise outdoors when AQI is 200?",
        "reference_answer": "AQI 200 falls in the 'Poor' category. Outdoor exercise should be reduced, especially for sensitive groups. Use N95 masks, prefer indoor exercise, and avoid prolonged exertion.",
        "expected_topics": ["exercise_outdoor", "health"],
    },
    {
        "question": "What is GRAP and what are its stages?",
        "reference_answer": "GRAP (Graded Response Action Plan) is Delhi's escalating emergency response to pollution. Stage I (Poor AQI 201-300): restrict dusty activities. Stage II (Very Poor 301-400): restrict diesel generators. Stage III (Severe 401-450): ban construction. Stage IV (Severe+ >450): ban entry of trucks, school closures.",
        "expected_topics": ["policy"],
    },
    {
        "question": "Compare Delhi's AQI monitoring with WHO guidelines",
        "reference_answer": "WHO 2021 guidelines recommend PM2.5 annual mean of 5 µg/m³ and 24-hour mean of 15 µg/m³. Delhi's PM2.5 levels regularly exceed these by 10-20x. India's NAQI has different breakpoints than WHO or US EPA.",
        "expected_topics": ["aqi_methodology", "comparison"],
    },
    {
        "question": "How has Delhi's air quality changed over the last decade?",
        "reference_answer": "Delhi has seen mixed trends. PM2.5 has slightly decreased due to interventions like BS-VI fuel, odd-even, and GRAP, but remains far above safe levels. Winter episodes remain severe. Some monitoring stations show improvement while others don't.",
        "expected_topics": ["historical_data"],
    },
]


# ── Metric functions ─────────────────────────────────────────────────

def faithfulness_score(answer: str, context_texts: list[str]) -> float:
    """Measure how much of the answer is grounded in the retrieved context.

    Uses sentence-level overlap: what fraction of answer sentences
    can be supported by context?
    """
    if not context_texts or not answer:
        return 0.0

    answer_sents = _split_sents(answer)
    if not answer_sents:
        return 0.0

    context_combined = " ".join(context_texts).lower()
    context_words = set(re.findall(r'\w+', context_combined))

    grounded = 0
    for sent in answer_sents:
        sent_words = set(re.findall(r'\w+', sent.lower()))
        # Remove stopwords
        sent_words -= _STOP
        if not sent_words:
            grounded += 1
            continue
        overlap = len(sent_words & context_words) / len(sent_words)
        if overlap > 0.4:  # at least 40% word overlap with context
            grounded += 1

    return grounded / len(answer_sents)


def answer_relevancy_score(answer: str, question: str) -> float:
    """Measure how relevant the answer is to the question.

    Uses keyword overlap heuristic.
    """
    if not answer or not question:
        return 0.0

    q_words = set(re.findall(r'\w+', question.lower())) - _STOP
    a_words = set(re.findall(r'\w+', answer.lower())) - _STOP

    if not q_words:
        return 0.5

    overlap = len(q_words & a_words) / len(q_words)
    # Bonus for substantial answers
    length_bonus = min(len(answer.split()) / 100, 0.2)

    return min(overlap + length_bonus, 1.0)


def context_precision_score(
    retrieved_chunks: list[dict],
    question: str,
    expected_topics: list[str] | None = None,
) -> float:
    """Measure if retrieved context is relevant to the query."""
    if not retrieved_chunks:
        return 0.0

    q_words = set(re.findall(r'\w+', question.lower())) - _STOP

    relevant = 0
    for chunk in retrieved_chunks:
        text = chunk.get("text", "").lower()
        chunk_words = set(re.findall(r'\w+', text))
        overlap = len(q_words & chunk_words) / max(len(q_words), 1)

        # Topic match bonus
        topic_match = False
        if expected_topics and chunk.get("topic") in expected_topics:
            topic_match = True

        if overlap > 0.3 or topic_match:
            relevant += 1

    return relevant / len(retrieved_chunks)


def context_recall_score(
    context_texts: list[str],
    reference_answer: str,
) -> float:
    """Does the context contain the info needed to produce the reference answer?"""
    if not context_texts or not reference_answer:
        return 0.0

    ref_sents = _split_sents(reference_answer)
    if not ref_sents:
        return 0.0

    context_combined = " ".join(context_texts).lower()
    context_words = set(re.findall(r'\w+', context_combined))

    covered = 0
    for sent in ref_sents:
        sent_words = set(re.findall(r'\w+', sent.lower())) - _STOP
        if not sent_words:
            covered += 1
            continue
        overlap = len(sent_words & context_words) / len(sent_words)
        if overlap > 0.35:
            covered += 1

    return covered / len(ref_sents)


def answer_correctness_score(answer: str, reference_answer: str) -> float:
    """F1-like score comparing answer to reference answer."""
    if not answer or not reference_answer:
        return 0.0

    a_words = set(re.findall(r'\w+', answer.lower())) - _STOP
    r_words = set(re.findall(r'\w+', reference_answer.lower())) - _STOP

    if not r_words or not a_words:
        return 0.0

    tp = len(a_words & r_words)
    precision = tp / len(a_words) if a_words else 0
    recall = tp / len(r_words) if r_words else 0

    if precision + recall == 0:
        return 0.0

    f1 = 2 * precision * recall / (precision + recall)
    return f1


# ── helpers ──────────────────────────────────────────────────────────

_STOP = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "can", "shall", "in", "on", "at", "to",
    "for", "of", "with", "and", "or", "but", "not", "no", "it", "its",
    "this", "that", "these", "those", "from", "by", "as", "if", "then",
    "than", "so", "up", "out", "about", "into", "over", "after", "what",
    "how", "why", "when", "where", "which", "who", "whom", "each",
    "every", "all", "both", "few", "more", "most", "other", "some",
}


def _split_sents(text: str) -> list[str]:
    sents = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sents if len(s.strip()) > 10]


# ── main evaluation runner ──────────────────────────────────────────

def run_evaluation(test_set: list[dict] | None = None, verbose: bool = True) -> dict:
    """Run full RAGAS-style evaluation on the RAG system.

    Returns aggregate metrics + per-question details.
    """
    from rag.retriever import Retriever
    from rag.query_classifier import classify_query
    from rag.llm_pipeline import LLMPipeline
    from rag.prompt_template import build_prompt

    test_set = test_set or DEFAULT_TEST_SET

    retriever = Retriever("embeddings/vector_store")
    llm = LLMPipeline()

    results = []
    totals = {
        "faithfulness": 0, "answer_relevancy": 0,
        "context_precision": 0, "context_recall": 0,
        "answer_correctness": 0,
    }

    for i, tc in enumerate(test_set):
        question = tc["question"]
        reference = tc["reference_answer"]
        expected_topics = tc.get("expected_topics", [])

        if verbose:
            print(f"\n{'='*60}")
            print(f"[{i+1}/{len(test_set)}] {question}")

        # Classify intent
        intent = classify_query(question)

        # Retrieve
        retrieved = retriever.retrieve(
            question, top_k=5,
            intent_config=intent.get("config"),
        )
        context_texts = [r.get("text", "") for r in retrieved]

        # Generate answer
        prompt = build_prompt(question, retrieved, {}, intent=intent)
        answer = llm.generate(prompt, max_length=1024)

        # Compute metrics
        faith = faithfulness_score(answer, context_texts)
        relevancy = answer_relevancy_score(answer, question)
        precision = context_precision_score(retrieved, question, expected_topics)
        recall = context_recall_score(context_texts, reference)
        correctness = answer_correctness_score(answer, reference)

        metrics = {
            "faithfulness": round(faith, 3),
            "answer_relevancy": round(relevancy, 3),
            "context_precision": round(precision, 3),
            "context_recall": round(recall, 3),
            "answer_correctness": round(correctness, 3),
        }

        for k, v in metrics.items():
            totals[k] += v

        result = {
            "question": question,
            "intent": intent["intent"],
            "num_retrieved": len(retrieved),
            "retrieved_topics": [r.get("topic", "?") for r in retrieved],
            "answer_preview": answer[:200],
            "metrics": metrics,
        }
        results.append(result)

        if verbose:
            print(f"  Intent: {intent['intent']} ({intent['confidence']:.0%})")
            print(f"  Retrieved: {len(retrieved)} chunks — topics: {result['retrieved_topics']}")
            for k, v in metrics.items():
                bar = "█" * int(v * 20) + "░" * (20 - int(v * 20))
                print(f"  {k:>20s}: {bar} {v:.3f}")

    # Aggregate
    n = len(test_set)
    aggregate = {k: round(v / n, 3) for k, v in totals.items()}
    overall = round(sum(aggregate.values()) / len(aggregate), 3)

    if verbose:
        print(f"\n{'='*60}")
        print("AGGREGATE SCORES")
        print(f"{'='*60}")
        for k, v in aggregate.items():
            bar = "█" * int(v * 20) + "░" * (20 - int(v * 20))
            print(f"  {k:>20s}: {bar} {v:.3f}")
        print(f"  {'OVERALL':>20s}: {'█' * int(overall * 20)}{'░' * (20 - int(overall * 20))} {overall:.3f}")

    return {
        "aggregate": aggregate,
        "overall": overall,
        "per_question": results,
        "num_questions": n,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="RAGAS evaluation for AQI RAG system")
    parser.add_argument("--test-file", help="JSON file with test cases")
    parser.add_argument("--quick", action="store_true", help="Run only 3 test cases")
    parser.add_argument("--output", help="Save results to JSON file")
    args = parser.parse_args()

    test_set = None
    if args.test_file:
        with open(args.test_file) as f:
            test_set = json.load(f)
    elif args.quick:
        test_set = DEFAULT_TEST_SET[:3]

    results = run_evaluation(test_set)

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")
