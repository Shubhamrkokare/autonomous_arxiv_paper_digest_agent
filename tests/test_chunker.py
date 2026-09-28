from pathlib import Path

import pytest

from arxiv_digest.models import ParsedPage
from arxiv_digest.services.chunker import chunk_pages, save_chunks


def test_chunker_keeps_page_and_section_metadata(tmp_path: Path) -> None:
    pages = [
        ParsedPage(
            page_number=1,
            text="Abstract\nA concise abstract about retrieval.\n1 Introduction\n" + "intro " * 12,
        ),
        ParsedPage(page_number=2, text="continued " * 8 + "\nReferences\n[1] A useful citation."),
    ]

    chunks = chunk_pages(
        pages,
        paper_id="2401.12345v1",
        chunk_size_words=8,
        chunk_overlap_words=2,
    )

    assert chunks[0].section == "Abstract"
    assert any(chunk.section == "1 Introduction" for chunk in chunks)
    assert any(chunk.section == "References" for chunk in chunks)
    assert all(chunk.page_number in {1, 2} for chunk in chunks)
    assert len({chunk.chunk_id for chunk in chunks}) == len(chunks)

    output = save_chunks(chunks, tmp_path / "chunks.json")
    assert '"page_number"' in output.read_text(encoding="utf-8")


def test_chunker_rejects_invalid_overlap() -> None:
    with pytest.raises(ValueError, match="smaller than chunk size"):
        chunk_pages(
            [ParsedPage(page_number=1, text="text")],
            paper_id="paper",
            chunk_size_words=10,
            chunk_overlap_words=10,
        )


def test_chunker_recombines_split_numbered_headings() -> None:
    chunks = chunk_pages(
        [
            ParsedPage(
                page_number=3,
                text="3.1\nEncoder and Decoder Stacks\nThe encoder has layers.",
            )
        ],
        paper_id="paper",
        chunk_size_words=20,
        chunk_overlap_words=2,
    )

    assert chunks[0].section == "3.1 Encoder and Decoder Stacks"
