"""PDF -> páginas -> chunks con metadatos (documento, título, página).

Decisiones:
- Extracción por página con pypdf para poder citar documento y página.
- Los chunks no cruzan páginas: cada cita apunta a una página real.
- Troceado por caracteres (~1.200) con corte preferente en fin de frase y
  solapamiento de 200 para no partir ideas entre chunks.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path

from langchain_core.documents import Document
from pypdf import PdfReader

from .config import CHUNK_OVERLAP, CHUNK_SIZE, DOCS_DIR

logging.getLogger("pypdf").setLevel(logging.ERROR)  # avisos de /MediaBox duplicado en PDFs de MDPI

_HYPHEN_BREAK = re.compile(r"(\w)-\n(\w)")
_MULTI_SPACE = re.compile(r"[ \t]+")
_MULTI_NEWLINE = re.compile(r"\n{2,}")
_SENTENCE_END = re.compile(r"[.!?]\s")
_AUTHOR_LINE = re.compile(r"(\*|@|\d{4}|;|,\s+[A-Z][\w.-]+\s+[A-Z])")
_SKIP_LINE = re.compile(r"^(sensors|article|review|citation|www\.|http|doi|received|academic editor|original research|published)", re.I)


def clean_text(text: str) -> str:
    text = _HYPHEN_BREAK.sub(r"\1\2", text)          # "measure-\nment" -> "measurement"
    text = text.replace("\r", "")
    text = _MULTI_SPACE.sub(" ", text)
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)      # saltos de línea simples -> espacio
    text = _MULTI_NEWLINE.sub("\n\n", text)
    return text.strip()


def guess_title(first_page_text: str, metadata_title: str | None = None) -> str:
    if metadata_title and len(metadata_title.strip()) >= 15 and not metadata_title.lower().startswith(("doi", "untitled")):
        return metadata_title.strip()
    lines = [ln.strip() for ln in first_page_text.split("\n") if ln.strip()]
    title_lines: list[str] = []
    for line in lines:
        if not title_lines:
            if len(line) >= 20 and not _SKIP_LINE.match(line) and not _AUTHOR_LINE.search(line):
                title_lines.append(line)
            continue
        if _AUTHOR_LINE.search(line) or len(line) < 10 or len(title_lines) >= 4:
            break
        title_lines.append(line)
    return " ".join(title_lines)


def split_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Ventanas de ~size caracteres, cortando en fin de frase si lo hay en el último 30 %."""
    chunks: list[str] = []
    start, n = 0, len(text)
    while start < n:
        end = min(start + size, n)
        if end < n:
            window = text[start + int(size * 0.7): end]
            cuts = [m.end() for m in _SENTENCE_END.finditer(window)]
            if cuts:
                end = start + int(size * 0.7) + cuts[-1]
        piece = text[start:end].strip()
        if len(piece) > 50:
            chunks.append(piece)
        if end >= n:
            break
        start = max(end - overlap, start + 1)
    return chunks


def load_documents(docs_dir: Path = DOCS_DIR) -> list[Document]:
    """Devuelve un Document de LangChain por chunk, con metadatos para citar."""
    docs: list[Document] = []
    for pdf in sorted(Path(docs_dir).glob("*.pdf")):
        reader = PdfReader(str(pdf))
        raw_pages = [p.extract_text() or "" for p in reader.pages]
        meta_title = reader.metadata.title if reader.metadata else None
        title = guess_title(raw_pages[0], meta_title) if raw_pages else pdf.stem
        for page_no, raw in enumerate(raw_pages, start=1):
            if not raw.strip():
                continue
            for j, piece in enumerate(split_text(clean_text(raw))):
                docs.append(Document(
                    page_content=piece,
                    metadata={"source": pdf.name, "title": title or pdf.stem, "page": page_no,
                              "chunk_id": f"{pdf.stem}:p{page_no}:c{j}"},
                ))
    return docs
