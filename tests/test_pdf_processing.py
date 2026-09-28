from pathlib import Path

import httpx
import pymupdf

from arxiv_digest.models import ParseQuality
from arxiv_digest.services.pdf_processing import download_pdf, extract_pages, save_parse_result


def create_pdf(path: Path, text: str) -> None:
    document = pymupdf.open()
    page = document.new_page()
    for line_number in range(25):
        page.insert_text((72, 72 + line_number * 18), text)
    document.save(path)
    document.close()


def create_pdf_bytes(text: str) -> bytes:
    document = pymupdf.open()
    page = document.new_page()
    page.insert_text((72, 72), text)
    pdf_bytes = document.tobytes()
    document.close()
    return pdf_bytes


def test_extract_pages_retains_page_numbers_and_marks_full_text(tmp_path: Path) -> None:
    pdf_path = tmp_path / "paper.pdf"
    create_pdf(pdf_path, "This is enough extracted paper text for a realistic parser test.")

    result = extract_pages(pdf_path, min_characters=100)

    assert result.quality is ParseQuality.FULL_TEXT
    assert result.pages[0].page_number == 1
    assert result.total_characters >= 100


def test_extract_pages_marks_short_text_for_abstract_fallback(tmp_path: Path) -> None:
    pdf_path = tmp_path / "short.pdf"
    create_pdf(pdf_path, "Short text")

    result = extract_pages(pdf_path, min_characters=10_000)

    assert result.quality is ParseQuality.ABSTRACT_ONLY
    assert "abstract fallback" in result.warnings[-1]


def test_download_pdf_accepts_valid_pdf_bytes_and_persists_parse_result(tmp_path: Path) -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=create_pdf_bytes("Downloaded paper text"),
            headers={"content-type": "application/pdf"},
        )

    destination = tmp_path / "paper.pdf"
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        downloaded_path = download_pdf(
            "https://arxiv.org/pdf/1234.5678",
            destination,
            max_size_bytes=1_000,
            client=client,
        )

    assert downloaded_path.read_bytes().startswith(b"%PDF-")

    result = extract_pages(downloaded_path, min_characters=1)
    saved_path = save_parse_result(result, tmp_path / "pages.json")
    assert '"quality"' in saved_path.read_text(encoding="utf-8")


def test_extract_pages_records_section_headings_and_references(tmp_path: Path) -> None:
    pdf_path = tmp_path / "structured.pdf"
    create_pdf(
        pdf_path,
        "1 Method\nThis is method text.\nReferences\n[1] A reference entry.",
    )

    result = extract_pages(pdf_path, min_characters=1)

    assert "1 Method" in result.section_headings
    assert "References" in result.section_headings
    assert result.references_text is not None
    assert "[1] A reference entry." in result.references_text
