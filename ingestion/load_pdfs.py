import pdfplumber
from pathlib import Path


def load_pdf_text(path: str) -> list:
    """Return list of page dicts with page texts and metadata."""
    pathp = Path(path)
    docs = []
    try:
        with pdfplumber.open(path) as pdf:
            for i, page in enumerate(pdf.pages, start=1):
                text = page.extract_text() or ""
                docs.append({
                    "doc_name": pathp.name,
                    "page": i,
                    "text": text,
                    "source_type": "pdf",
                })
    except Exception:
        # fallback: treat as plain text file
        try:
            with open(path, 'r', encoding='utf-8') as f:
                text = f.read()
            docs.append({
                "doc_name": pathp.name,
                "page": 1,
                "text": text,
                "source_type": "text",
            })
        except Exception:
            pass
    return docs


def load_all_pdfs(pdf_dir: str):
    pdf_dir = Path(pdf_dir)
    results = []
    # handle .pdf and .txt files
    for ext in ("*.pdf", "*.txt"):
        for p in sorted(pdf_dir.glob(ext)):
            results.extend(load_pdf_text(str(p)))
    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("pdf_dir", help="Directory with PDFs or text files")
    args = parser.parse_args()
    pages = load_all_pdfs(args.pdf_dir)
    print(f"Loaded {len(pages)} pages from docs")
