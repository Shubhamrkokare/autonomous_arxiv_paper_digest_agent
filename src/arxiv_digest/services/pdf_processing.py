"""Safe PDF download, page-aware extraction, and persisted parse artifacts."""

from pathlib import Path

import httpx
import pymupdf

from arxiv_digest.models import ParsedPage, ParseQuality, ParseResult


class PdfProcessingError(RuntimeError):
    """Raised when a paper PDF cannot safely be downloaded or opened."""


def download_pdf(
    pdf_url: str,
    destination: Path,
    *,
    max_size_bytes: int,
    client: httpx.Client | None = None,
) -> Path:
    """Download a PDF atomically while enforcing a caller-supplied size limit."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = destination.with_suffix(f"{destination.suffix}.part")
    active_client = client or httpx.Client(follow_redirects=True, timeout=30.0)
    should_close_client = client is None

    try:
        with active_client.stream(
            "GET", pdf_url, headers={"Accept": "application/pdf"}
        ) as response:
            response.raise_for_status()
            declared_size = response.headers.get("content-length")
            if declared_size and int(declared_size) > max_size_bytes:
                raise PdfProcessingError("The PDF exceeds the configured download-size limit.")

            content_type = response.headers.get("content-type", "").lower()
            if content_type and "pdf" not in content_type:
                raise PdfProcessingError("The download response is not a PDF document.")

            bytes_written = 0
            with temporary_path.open("wb") as file_handle:
                for block in response.iter_bytes():
                    bytes_written += len(block)
                    if bytes_written > max_size_bytes:
                        raise PdfProcessingError(
                            "The PDF exceeds the configured download-size limit."
                        )
                    file_handle.write(block)

        if temporary_path.read_bytes()[:5] != b"%PDF-":
            raise PdfProcessingError("The downloaded file does not have a valid PDF header.")
        temporary_path.replace(destination)
        return destination
    except (httpx.HTTPError, OSError) as error:
        raise PdfProcessingError(f"Could not download the paper PDF: {error}") from error
    finally:
        if temporary_path.exists():
            temporary_path.unlink()
        if should_close_client:
            active_client.close()


def extract_pages(pdf_path: Path, *, min_characters: int) -> ParseResult:
    """Extract normalized page text and signal when it is insufficient for full QA."""

    try:
        document = pymupdf.open(pdf_path)
    except (pymupdf.FileDataError, OSError, RuntimeError) as error:
        raise PdfProcessingError(f"Could not open the downloaded PDF: {error}") from error

    pages: list[ParsedPage] = []
    blank_pages = 0
    try:
        for page_number, page in enumerate(document, start=1):
            lines = [" ".join(line.split()) for line in page.get_text("text").splitlines()]
            text = "\n".join(line for line in lines if line)
            if text:
                pages.append(ParsedPage(page_number=page_number, text=text))
            else:
                blank_pages += 1
    finally:
        document.close()

    total_characters = sum(len(page.text) for page in pages)
    warnings: list[str] = []
    if blank_pages:
        warnings.append(f"{blank_pages} page(s) had no extractable text.")

    quality = ParseQuality.FULL_TEXT
    if total_characters < min_characters:
        quality = ParseQuality.ABSTRACT_ONLY
        warnings.append(
            "Extracted text is too short for reliable full-paper QA; "
            "use the arXiv abstract fallback."
        )

    return ParseResult(
        pdf_path=pdf_path,
        quality=quality,
        pages=pages,
        total_characters=total_characters,
        warnings=warnings,
    )


def save_parse_result(result: ParseResult, destination: Path) -> Path:
    """Persist the parse outcome as JSON for later graph nodes and resumed sessions."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(result.model_dump_json(indent=2), encoding="utf-8")
    return destination
