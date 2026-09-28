"""Section-aware, page-citable text chunking for paper retrieval."""

import re
from pathlib import Path

from arxiv_digest.models import ParsedPage, TextChunk

SECTION_PATTERN = re.compile(
    r"^(?:(?:\d+(?:\.\d+)*)|(?:[IVXLC]+))\.?\s+[A-Z][A-Za-z0-9 ,:;()\-]{1,110}$"
)
SECTION_NUMBER_PATTERN = re.compile(r"^(?:\d+(?:\.\d+)*|[IVXLC]+)\.?$")
NAMED_SECTION_HEADINGS = {
    "abstract",
    "acknowledgments",
    "appendix",
    "conclusion",
    "conclusions",
    "experiments",
    "methods",
    "references",
}


def _is_section_heading(line: str) -> bool:
    """Recognize common numbered and named academic-paper section headings."""

    normalized = " ".join(line.split())
    return normalized.lower() in NAMED_SECTION_HEADINGS or bool(
        SECTION_PATTERN.fullmatch(normalized)
    )


def _normalize_page_lines(text: str) -> list[str]:
    """Join PDF-extracted number-only heading lines to their title lines."""

    lines = [" ".join(line.split()) for line in text.splitlines()]
    cleaned_lines = [line for line in lines if line]
    normalized: list[str] = []
    index = 0
    while index < len(cleaned_lines):
        line = cleaned_lines[index]
        next_line = cleaned_lines[index + 1] if index + 1 < len(cleaned_lines) else ""
        if (
            SECTION_NUMBER_PATTERN.fullmatch(line)
            and next_line
            and next_line[0].isupper()
            and len(next_line) <= 112
        ):
            normalized.append(f"{line} {next_line}")
            index += 2
        else:
            normalized.append(line)
            index += 1
    return normalized


def _page_segments(page: ParsedPage, inherited_section: str) -> tuple[list[tuple[str, str]], str]:
    """Split a page at headings and return its final section for the next page."""

    current_section = inherited_section
    segments: list[tuple[str, str]] = []
    buffer: list[str] = []

    for cleaned in _normalize_page_lines(page.text):
        if _is_section_heading(cleaned):
            if buffer:
                segments.append((current_section, " ".join(buffer)))
            current_section = cleaned
            buffer = [cleaned]
        else:
            buffer.append(cleaned)

    if buffer:
        segments.append((current_section, " ".join(buffer)))
    return segments, current_section


def _window_words(words: list[str], size: int, overlap: int) -> list[list[str]]:
    """Split words into overlapping windows, validating configuration early."""

    if size <= 0:
        raise ValueError("Chunk size must be greater than zero.")
    if overlap < 0 or overlap >= size:
        raise ValueError("Chunk overlap must be non-negative and smaller than chunk size.")

    step = size - overlap
    return [
        words[start : start + size]
        for start in range(0, len(words), step)
        if words[start : start + size]
    ]


def chunk_pages(
    pages: list[ParsedPage],
    *,
    paper_id: str,
    chunk_size_words: int,
    chunk_overlap_words: int,
) -> list[TextChunk]:
    """Create page-citable chunks while carrying section context across pages."""

    chunks: list[TextChunk] = []
    current_section = "Front matter"
    chunk_number = 1

    for page in pages:
        segments, current_section = _page_segments(page, current_section)
        for section, segment in segments:
            for word_window in _window_words(
                segment.split(), chunk_size_words, chunk_overlap_words
            ):
                chunks.append(
                    TextChunk(
                        chunk_id=f"{paper_id}_p{page.page_number}_c{chunk_number}",
                        paper_id=paper_id,
                        page_number=page.page_number,
                        section=section,
                        text=" ".join(word_window),
                        word_count=len(word_window),
                    )
                )
                chunk_number += 1
    return chunks


def save_chunks(chunks: list[TextChunk], destination: Path) -> Path:
    """Persist chunks for inspection and future session-resume support."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    serialized = "[\n" + ",\n".join(chunk.model_dump_json() for chunk in chunks) + "\n]\n"
    destination.write_text(serialized, encoding="utf-8")
    return destination
