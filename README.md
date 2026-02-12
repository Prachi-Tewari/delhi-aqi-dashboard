# RAG-Based Delhi AQI Analysis & Explanation System

An end-to-end AI system that combines **Retrieval-Augmented Generation (RAG)** over Delhi AQI research papers with **real-time pollutant data** from OpenAQ to produce data-driven analysis, charts, and layman-friendly explanations.

## Features

- **Document Ingestion** — Load PDFs / text files, chunk, embed with `sentence-transformers/all-MiniLM-L6-v2`, store in FAISS
- **Real-Time Data** — Fetches PM2.5, PM10, NO₂, SO₂, CO, O₃ from the OpenAQ v2 API (with demo-data fallback)
- **RAG Pipeline** — Retrieves relevant document chunks + injects live data into the LLM prompt
- **Open-Source LLM** — Default `google/flan-t5-base` (local CPU); also supports causal models (Mistral, Llama) and HuggingFace Inference API
- **Visualization** — Time-series charts, pollutant bar plots, AQI category cards (matplotlib)
- **Streamlit UI** — City selector, question box, charts + AI explanation in one page

## Quick Start

```bash
# 1. Create & activate a virtual environment
python -m venv .venv && source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Place AQI PDFs / reports into data/pdfs/

# 4. Build the vector store (embeddings)
python -m ingestion.embed_docs --pdf_dir data/pdfs --out_dir embeddings/vector_store

# 5. Launch the app
streamlit run app.py
```

## Environment Variables (optional)

| Variable | Purpose |
|---|---|
| `OPENAQ_API_KEY` | OpenAQ API key — avoids rate limits, enables full data access |
| `HF_TOKEN` | HuggingFace token — required only when using the Inference API |

```bash
export OPENAQ_API_KEY="your_key_here"
export HF_TOKEN="hf_..."
```

## Folder Structure

```
aqi_rag_system/
├── api/
│   └── openaq_client.py      # OpenAQ v2 API client + demo fallback
├── data/
│   ├── pdfs/                  # Drop your AQI PDFs here
│   └── processed_docs/
├── embeddings/
│   ├── vector_store.py        # FAISS / sklearn vector store
│   └── vector_store/          # Saved index + metadata
├── ingestion/
│   ├── load_pdfs.py           # PDF / text loader
│   ├── chunk_docs.py          # Sentence-aware text chunker
│   └── embed_docs.py          # Embedding builder (sentence-transformers / TF-IDF)
├── rag/
│   ├── retriever.py           # Query-time retrieval
│   ├── prompt_template.py     # Prompt assembly
│   └── llm_pipeline.py        # LLM wrapper (local + API)
├── visualization/
│   └── plots.py               # Matplotlib chart generators
├── app.py                     # Streamlit UI
├── requirements.txt
└── README.md
```

## Notes

- The system falls back to **synthetic demo data** when the OpenAQ API is unreachable, so the app always runs.
- The default LLM (`flan-t5-base`, 250 MB) runs on CPU. For better answers, try `google/flan-t5-large` or enable the HuggingFace Inference API in the sidebar.
- Embeddings use `sentence-transformers/all-MiniLM-L6-v2`; a TF-IDF fallback is used automatically if the package is missing.
