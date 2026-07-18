"""
pdf_agent.py
------------
Responsibilities (per architecture doc):
    - Upload PDF
    - Extract text
    - OCR for scanned pages
    - Clean text
    - Split into chunks
    - Store in Vector DB
"""

import re
from typing import List, Dict

import fitz  # PyMuPDF

from database.vector_store import SimpleVectorStore


class PDFAgent:
    def __init__(self, vector_store: SimpleVectorStore, chunk_size: int = 1000, overlap: int = 150):
        self.store = vector_store
        self.chunk_size = chunk_size
        self.overlap = overlap

    # ------------------------------------------------------------------ #
    def process(self, pdf_path: str, doc_id: str) -> Dict:
        raw_text = self._extract_text(pdf_path)
        cleaned = self._clean(raw_text)
        chunks = self._chunk(cleaned)

        for i, chunk in enumerate(chunks):
            self.store.add(
                doc_id=f"{doc_id}::chunk_{i}",
                text=chunk,
                meta={"source_doc": doc_id, "chunk_index": i},
            )

        return {
            "doc_id": doc_id,
            "num_pages_processed": self._page_count(pdf_path),
            "num_chunks": len(chunks),
            "char_count": len(cleaned),
        }

    # ------------------------------------------------------------------ #
    def _page_count(self, pdf_path: str) -> int:
        with fitz.open(pdf_path) as doc:
            return doc.page_count

    def _extract_text(self, pdf_path: str) -> str:
        text_parts = []
        with fitz.open(pdf_path) as doc:
            for page in doc:
                page_text = page.get_text().strip()
                if not page_text:
                    # Scanned / image-only page -> try OCR
                    page_text = self._ocr_page(page)
                text_parts.append(page_text)
        return "\n\n".join(text_parts)

    def _ocr_page(self, page) -> str:
        """OCR fallback for scanned pages. Requires pytesseract + Pillow."""
        try:
            import pytesseract
            from PIL import Image
            import io

            pix = page.get_pixmap(dpi=200)
            img = Image.open(io.BytesIO(pix.tobytes("png")))
            return pytesseract.image_to_string(img)
        except Exception as e:
            return f"[OCR unavailable or failed: {e}]"

    def _clean(self, text: str) -> str:
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    def _chunk(self, text: str) -> List[str]:
        if not text:
            return []
        chunks = []
        start = 0
        n = len(text)
        while start < n:
            end = min(start + self.chunk_size, n)
            chunks.append(text[start:end])
            if end == n:
                break
            start = end - self.overlap
        return chunks
