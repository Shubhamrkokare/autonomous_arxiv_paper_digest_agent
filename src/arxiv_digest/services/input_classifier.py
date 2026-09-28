"""Classify the assessment's supported arXiv ID, URL, and topic inputs."""

import re
from urllib.parse import urlparse

from arxiv_digest.models import InputKind, QueryInput


class InputValidationError(ValueError):
    """Raised when input is empty or looks like an unsupported paper URL."""


# New-style IDs: 2401.12345 or 2401.12345v2. Legacy IDs are also accepted:
# cs/0112017 and cs/0112017v3.
ARXIV_ID_PATTERN = re.compile(
    r"^(?:(?P<modern>\d{4}\.\d{4,5})|(?P<legacy>[a-z-]+(?:\.[a-z-]+)?/\d{7}))(?:v\d+)?$",
    re.IGNORECASE,
)
ARXIV_URL_PATTERN = re.compile(
    r"^/(?:abs|pdf)/(?P<arxiv_id>(?:(?:\d{4}\.\d{4,5})|(?:[a-z-]+(?:\.[a-z-]+)?/\d{7}))(?:v\d+)?)(?:\.pdf)?/?$",
    re.IGNORECASE,
)
ARXIV_HOSTS = {"arxiv.org", "www.arxiv.org", "export.arxiv.org"}


def _normalize_id(candidate: str) -> str | None:
    """Return a validated, normalized ID while retaining an explicit version."""

    normalized = candidate.strip().removeprefix("arXiv:").strip()
    if ARXIV_ID_PATTERN.fullmatch(normalized):
        return normalized.lower()
    return None


def classify_input(raw_input: str) -> QueryInput:
    """Recognize an arXiv identifier, an arXiv URL, or a topic-search query.

    A non-arXiv URL is rejected instead of being searched as a topic. This gives
    users an actionable error and avoids silently interpreting a pasted URL as
    unrelated research keywords.
    """

    cleaned = raw_input.strip()
    if not cleaned:
        raise InputValidationError("Please provide an arXiv ID, an arXiv URL, or a research topic.")

    direct_id = _normalize_id(cleaned)
    if direct_id:
        return QueryInput(raw_input=raw_input, kind=InputKind.ARXIV_ID, arxiv_id=direct_id)

    parsed_url = urlparse(cleaned)
    if parsed_url.scheme and parsed_url.netloc:
        if parsed_url.netloc.lower() not in ARXIV_HOSTS:
            raise InputValidationError(
                "Only arXiv URLs are supported; provide a research topic instead."
            )

        match = ARXIV_URL_PATTERN.fullmatch(parsed_url.path)
        if not match:
            raise InputValidationError(
                "That arXiv URL does not contain a valid /abs/ or /pdf/ paper ID."
            )

        arxiv_id = _normalize_id(match.group("arxiv_id"))
        if arxiv_id is None:  # Defensive guard if the two patterns diverge later.
            raise InputValidationError("The arXiv ID in this URL is invalid.")
        return QueryInput(raw_input=raw_input, kind=InputKind.ARXIV_ID, arxiv_id=arxiv_id)

    if "://" in cleaned:
        raise InputValidationError(
            "Only arXiv URLs are supported; provide a research topic instead."
        )

    return QueryInput(raw_input=raw_input, kind=InputKind.TOPIC, topic=cleaned)
