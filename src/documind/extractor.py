"""Document text and metadata extraction plus overlap-aware chunking."""

from __future__ import annotations

import io
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pypdf import PdfReader

_TEXT_EXTS = {".txt", ".md", ".markdown", ".rst", ".csv", ".json", ".py", ".log", ".yaml", ".yml"}
_WORD_RE = re.compile(r"[a-zA-Z0-9_]+")


class ExtractionError(ValueError):
    """Raised when a document cannot be parsed."""


@dataclass(slots=True)
class ExtractedDocument:
    filename: str
    mime_type: str
    title: str
    author: str
    page_count: int
    pages: list[tuple[int, str]]  # (page_number, text)


def extract_document(filename: str, content: bytes) -> ExtractedDocument:
    ext = Path(filename).suffix.lower()
    if ext == ".pdf":
        try:
            reader = PdfReader(io.BytesIO(content))
        except Exception as exc:
            raise ExtractionError(f"Failed to parse PDF: {exc}") from exc
        meta: Any = reader.metadata or {}
        title = str(meta.get("/Title") or Path(filename).stem)
        author = str(meta.get("/Author") or "")
        pages: list[tuple[int, str]] = []
        for idx, page in enumerate(reader.pages, start=1):
            txt = (page.extract_text() or "").strip()
            if txt:
                pages.append((idx, txt))
        if not pages:
            raise ExtractionError("PDF contains no extractable text")
        return ExtractedDocument(
            filename=Path(filename).name,
            mime_type="application/pdf",
            title=title,
            author=author,
            page_count=len(reader.pages),
            pages=pages,
        )

    if ext in _TEXT_EXTS:
        if b"\x00" in content[:4096]:
            raise ExtractionError("Binary content detected in text file")
        try:
            text = content.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ExtractionError("Text document must be UTF-8 encoded") from exc
        title = Path(filename).stem
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith("# "):
                title = stripped[2:].strip()[:120]
                break
        return ExtractedDocument(
            filename=Path(filename).name,
            mime_type="text/plain",
            title=title,
            author="",
            page_count=1,
            pages=[(1, text)],
        )

    raise ExtractionError(f"Unsupported document extension '{ext or '(none)'}'")


def chunk_pages(
    pages: list[tuple[int, str]],
    chunk_size: int = 700,
    overlap: int = 100,
) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    step = max(50, chunk_size - overlap)
    chunk_idx = 0
    for page_num, raw_text in pages:
        text = " ".join(raw_text.split())
        if not text:
            continue
        start = 0
        while start < len(text):
            piece = text[start : start + chunk_size].strip()
            if piece:
                chunks.append({"chunk_index": chunk_idx, "page_number": page_num, "content": piece})
                chunk_idx += 1
            if start + chunk_size >= len(text):
                break
            start += step
    return chunks


def tokenize(text: str) -> list[str]:
    return [m.group(0).lower() for m in _WORD_RE.finditer(text)]
