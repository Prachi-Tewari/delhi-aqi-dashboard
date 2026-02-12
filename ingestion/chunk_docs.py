"""Semantic chunking with token-aware overlap.

Instead of splitting on a fixed character count, this module:
  1. Splits text into sentences.
  2. Embeds each sentence with a lightweight model.
  3. Detects *semantic breakpoints* where consecutive-sentence similarity
     drops below a threshold — those become chunk boundaries.
  4. Adds a 200-token overlap between adjacent chunks so context carries
     forward.

Falls back to a simpler paragraph-aware splitter when sentence-transformers
is not available or the text is very short.
"""

from __future__ import annotations

import re
import math
from typing import List
from collections import Counter


# ── helpers ──────────────────────────────────────────────────────────

def _split_sentences(text: str) -> list[str]:
    """Split text into sentences (handles abbreviations reasonably)."""
    text = text.replace("\r", "")
    parts = re.split(r'(?<=[.!?])\s+|\n{2,}', text)
    return [s.strip() for s in parts if s.strip()]


def _approx_tokens(text: str) -> int:
    """Rough token count (≈ words × 1.3)."""
    return max(1, int(len(text.split()) * 1.3))


def _get_sentence_embedder():
    """Return a lightweight embedding function or None."""
    try:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer("all-MiniLM-L6-v2")
        return lambda texts: _model.encode(texts, convert_to_numpy=True,
                                           show_progress_bar=False)
    except Exception:
        return None


def _cosine_sim(a, b):
    import numpy as np
    dot = float(np.dot(a, b))
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    return dot / (na * nb + 1e-9)


# ── core semantic chunker ────────────────────────────────────────────

def semantic_chunk(
    text: str,
    max_tokens: int = 512,
    overlap_tokens: int = 200,
    similarity_threshold: float = 0.45,
) -> list[str]:
    """Split *text* into semantically coherent chunks with overlap."""
    sentences = _split_sentences(text)
    if not sentences:
        return []
    if len(sentences) <= 3:
        return [text.strip()] if text.strip() else []

    embed_fn = _get_sentence_embedder()
    if embed_fn is not None and len(sentences) > 5:
        return _semantic_split(sentences, embed_fn, max_tokens,
                               overlap_tokens, similarity_threshold)
    else:
        return _fallback_split(sentences, max_tokens, overlap_tokens)


def _semantic_split(sentences, embed_fn, max_tokens, overlap_tokens, threshold):
    """Use embedding similarity to find natural breakpoints."""
    embs = embed_fn(sentences)
    sims = [_cosine_sim(embs[i], embs[i + 1]) for i in range(len(sentences) - 1)]
    breakpoints = {i + 1 for i, s in enumerate(sims) if s < threshold}

    groups: list[list[str]] = []
    current: list[str] = []
    current_tok = 0
    for i, sent in enumerate(sentences):
        stok = _approx_tokens(sent)
        if current and (i in breakpoints or current_tok + stok > max_tokens):
            groups.append(current)
            overlap_sents = _get_overlap(current, overlap_tokens)
            current = overlap_sents + [sent]
            current_tok = sum(_approx_tokens(s) for s in current)
        else:
            current.append(sent)
            current_tok += stok
    if current:
        groups.append(current)
    return [" ".join(g) for g in groups]


def _fallback_split(sentences, max_tokens, overlap_tokens):
    """Simple sentence-based splitting when embeddings are unavailable."""
    groups: list[list[str]] = []
    current: list[str] = []
    current_tok = 0
    for sent in sentences:
        stok = _approx_tokens(sent)
        if current and current_tok + stok > max_tokens:
            groups.append(current)
            overlap_sents = _get_overlap(current, overlap_tokens)
            current = overlap_sents + [sent]
            current_tok = sum(_approx_tokens(s) for s in current)
        else:
            current.append(sent)
            current_tok += stok
    if current:
        groups.append(current)
    return [" ".join(g) for g in groups]


def _get_overlap(sentences: list[str], overlap_tokens: int) -> list[str]:
    """Return the last N sentences that fit in overlap_tokens."""
    result: list[str] = []
    tok = 0
    for s in reversed(sentences):
        stok = _approx_tokens(s)
        if tok + stok > overlap_tokens:
            break
        result.insert(0, s)
        tok += stok
    return result


# ── metadata helpers ─────────────────────────────────────────────────

def _extract_year(text: str):
    years = re.findall(r'\b(20[0-2]\d|19[89]\d)\b', text)
    if years:
        return int(Counter(years).most_common(1)[0][0])
    return None


def _infer_topic(text: str) -> str:
    text_lower = text.lower()
    scores = {
        "health": len(re.findall(
            r'health|disease|asthma|lung|respiratory|mortality|hospital|medical|copd', text_lower)),
        "policy": len(re.findall(
            r'policy|government|regulation|grap|ngt|cpcb|ministry|ban|guideline|standard', text_lower)),
        "pollution_sources": len(re.findall(
            r'stubble|vehicle|transport|industrial|emission|burning|construction|dust', text_lower)),
        "aqi_methodology": len(re.findall(
            r'aqi|index|breakpoint|sub.index|naqi|epa|who|methodology|calculation', text_lower)),
        "meteorology": len(re.findall(
            r'wind|temperature|inversion|humidity|monsoon|weather|geography|climate', text_lower)),
        "historical_data": len(re.findall(
            r'\d{4}|trend|annual|seasonal|winter|summer|year|decade|historical', text_lower)),
        "exercise_outdoor": len(re.findall(
            r'exercise|jog|run|outdoor|mask|purifier|activity|fitness|sport', text_lower)),
    }
    if not any(scores.values()):
        return "general"
    return max(scores, key=scores.get)


def _credibility_score(doc_name: str, source_type: str) -> float:
    name_lower = (doc_name or "").lower()
    if any(kw in name_lower for kw in ["research", "study", "paper", "journal", "pubmed"]):
        return 0.95
    if any(kw in name_lower for kw in ["government", "cpcb", "who", "epa", "official"]):
        return 0.92
    if any(kw in name_lower for kw in ["comprehensive", "methodology", "guide"]):
        return 0.88
    if source_type == "pdf":
        return 0.85
    if any(kw in name_lower for kw in ["historical", "data"]):
        return 0.82
    return 0.75


# ── document-level API (replaces old chunk_texts) ────────────────────

def chunk_texts(pages: List[dict], max_tokens: int = 512,
                overlap_tokens: int = 200) -> list[dict]:
    """Semantically chunk pages and attach rich metadata.

    Each returned dict has:
        doc_name, page, source_type, text,
        topic, year, credibility_score, token_count
    """
    chunks = []
    for p in pages:
        text = p.get("text", "")
        if not text or len(text.strip()) < 40:
            continue
        doc_name = p.get("doc_name", "unknown")
        source_type = p.get("source_type", "text")
        raw_chunks = semantic_chunk(text, max_tokens=max_tokens,
                                    overlap_tokens=overlap_tokens)
        for ci, chunk_text in enumerate(raw_chunks):
            chunks.append({
                "doc_name": doc_name,
                "page": p.get("page", ci + 1),
                "source_type": source_type,
                "text": chunk_text,
                "topic": _infer_topic(chunk_text),
                "year": _extract_year(chunk_text),
                "credibility_score": _credibility_score(doc_name, source_type),
                "token_count": _approx_tokens(chunk_text),
            })
    return chunks


if __name__ == "__main__":
    import json
    import argparse
    from ingestion.load_pdfs import load_all_pdfs

    parser = argparse.ArgumentParser()
    parser.add_argument("pdf_dir")
    parser.add_argument("out_file")
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--overlap", type=int, default=200)
    args = parser.parse_args()

    pages = load_all_pdfs(args.pdf_dir)
    chunks = chunk_texts(pages, max_tokens=args.max_tokens,
                         overlap_tokens=args.overlap)
    with open(args.out_file, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    print(f"Created {len(chunks)} semantic chunks")
    topics = {}
    for c in chunks:
        t = c.get("topic", "general")
        topics[t] = topics.get(t, 0) + 1
    print(f"Topics: {topics}")
    print(f"Wrote {len(chunks)} chunks to {args.out_file}")
