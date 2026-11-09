import hashlib
import re
from typing import Iterator, List, Optional, Tuple

import pdfplumber
from langchain_community.document_loaders import Docx2txtLoader, TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from utils.config import settings

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt"}

# Changing this makes re-uploads of existing files get processed again.
INGEST_VERSION = "2"

# Lower than pdfplumber's default so that gaps next to bold text stay as spaces.
PDF_X_TOLERANCE = 1.5

NUMBERED_HEADING = re.compile(r"^\d+(\.\d+)*\.?\s+[A-Z]")

# (page, text line or None, table rows or None)
Item = Tuple[int, Optional[str], Optional[List[List[str]]]]


def get_document_hash(file_path: str) -> str:
    hasher = hashlib.sha256(INGEST_VERSION.encode())
    with open(file_path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def _pdf_page_items(page, number: int) -> List[Item]:
    tables = [t for t in page.find_tables() if len(t.rows) >= 2 and len(t.rows[0].cells) >= 2]
    boxes = [t.bbox for t in tables]

    def outside_tables(obj) -> bool:
        x, y = (obj["x0"] + obj["x1"]) / 2, (obj["top"] + obj["bottom"]) / 2
        return not any(x0 <= x <= x1 and top <= y <= bottom for x0, top, x1, bottom in boxes)

    placed = []
    for line in page.filter(outside_tables).extract_text_lines(x_tolerance=PDF_X_TOLERANCE):
        placed.append((line["top"], (number, line["text"], None)))
    for table in tables:
        rows = [[" ".join((cell or "").split()) for cell in row] for row in table.extract(x_tolerance=PDF_X_TOLERANCE)]
        placed.append((table.bbox[1], (number, None, rows)))
    placed.sort(key=lambda entry: entry[0])
    return [item for _, item in placed]


def _read_items(file_path: str, ext: str) -> Iterator[Item]:
    if ext == ".pdf":
        with pdfplumber.open(file_path) as pdf:
            for number, page in enumerate(pdf.pages):
                yield from _pdf_page_items(page, number)
        return

    if ext == ".docx":
        loader = Docx2txtLoader(file_path)
    elif ext == ".txt":
        loader = TextLoader(file_path, autodetect_encoding=True)
    else:
        raise ValueError(f"Unsupported file extension: {ext}")
    for doc in loader.load():
        for line in doc.page_content.split("\n"):
            yield 0, line, None


def _is_heading(line: str) -> bool:
    text = line.strip().rstrip(":")
    if not 2 <= len(text) <= 60 or text[-1] in ".;" or "," in text or "|" in text:
        return False
    letters = [c for c in text if c.isalpha()]
    return (len(letters) >= 3 and all(c.isupper() for c in letters)) or bool(NUMBERED_HEADING.match(text))


def _table_pieces(rows: List[List[str]]) -> List[str]:
    """Render a table as Markdown, repeating its header in every piece."""
    render = lambda row: "| " + " | ".join(row) + " |"
    header = render(rows[0]) + "\n" + render(["---"] * len(rows[0]))
    pieces, current = [], header
    for row in rows[1:]:
        line = render(row)
        if len(current) + len(line) + 1 > settings.CHUNK_SIZE and current != header:
            pieces.append(current)
            current = header
        current += "\n" + line
    pieces.append(current)
    return pieces


def process_document(file_path: str, ext: str, source: str, doc_path: str, role: str) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
    )
    doc_hash = get_document_hash(file_path)
    chunks: List[Document] = []
    section, buffer, buffer_page = "", [], 0

    def add(text: str, page: int):
        content = f"Section: {section}\n{text}" if section else text
        chunks.append(
            Document(
                page_content=content,
                metadata={
                    "source": source,
                    "doc_path": doc_path,
                    "doc_hash": doc_hash,
                    "role": role,
                    "page": page,
                    "section": section,
                    "chunk_index": len(chunks),
                },
            )
        )

    def flush():
        nonlocal buffer
        text = "\n".join(buffer).strip()
        buffer = []
        for piece in splitter.split_text(text):
            if piece.strip():
                add(piece, buffer_page)

    previous_table = None  # (page, header) of a table that may continue on the next page
    for page, line, rows in _read_items(file_path, ext):
        if rows is not None:
            flush()
            if previous_table:
                last_page, header = previous_table
                if page == last_page + 1 and len(rows[0]) == len(header) and rows[0] != header:
                    rows = [header] + rows
            previous_table = (page, rows[0])
            for piece in _table_pieces(rows):
                add(piece, page)
            continue
        if line.strip():
            previous_table = None
        if line.strip() and _is_heading(line):
            flush()
            section = line.strip().rstrip(":")
        else:
            if not buffer:
                buffer_page = page
            buffer.append(line)
    flush()
    return chunks
