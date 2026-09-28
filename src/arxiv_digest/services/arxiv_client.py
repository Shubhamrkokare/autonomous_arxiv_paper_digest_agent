"""A small, testable boundary around the official arXiv Atom API client."""

import re

import arxiv

from arxiv_digest.models import PaperMetadata


class ArxivRetrievalError(RuntimeError):
    """Raised when arXiv cannot return a usable paper for a valid request."""


TOPIC_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")
TOPIC_STOP_WORDS = {
    "a",
    "an",
    "and",
    "for",
    "from",
    "in",
    "of",
    "on",
    "recent",
    "the",
    "to",
    "with",
    "work",
}


def build_topic_query(topic: str) -> str:
    """Turn a natural-language topic into a conservative arXiv Boolean query.

    Each retained token is scoped with ``all:`` and combined with ``AND``. This
    avoids the API treating a sentence as a loose, undocumented free-text query.
    """

    terms = [
        token
        for token in TOPIC_TOKEN_PATTERN.findall(topic.lower())
        if token not in TOPIC_STOP_WORDS and len(token) >= 2
    ]
    if not terms:
        raise ArxivRetrievalError("The topic does not contain any searchable research terms.")
    return " AND ".join(f"all:{term}" for term in terms)


class ArxivService:
    """Retrieve and normalize arXiv records without exposing library objects."""

    def __init__(
        self,
        client: arxiv.Client | None = None,
        *,
        page_size: int = 10,
        delay_seconds: float = 3.0,
        num_retries: int = 2,
    ) -> None:
        self._client = client or arxiv.Client(
            page_size=page_size,
            delay_seconds=delay_seconds,
            num_retries=num_retries,
        )

    def lookup_paper(self, arxiv_id: str) -> PaperMetadata:
        """Fetch one exact paper by ID, or raise a clear domain error."""

        search = arxiv.Search(id_list=[arxiv_id], max_results=1)
        results = list(self._client.results(search))
        if not results:
            raise ArxivRetrievalError(f"No arXiv paper was found for ID '{arxiv_id}'.")
        return self._to_metadata(results[0])

    def search_topic(self, topic: str, *, limit: int = 10) -> list[PaperMetadata]:
        """Return relevant candidates; local ranking comes in the next stage."""

        search = arxiv.Search(
            query=build_topic_query(topic),
            max_results=limit,
            sort_by=arxiv.SortCriterion.Relevance,
            sort_order=arxiv.SortOrder.Descending,
        )
        return [self._to_metadata(result) for result in self._client.results(search)]

    @staticmethod
    def _to_metadata(result: arxiv.Result) -> PaperMetadata:
        """Map a third-party result to the stable model our application owns."""

        return PaperMetadata(
            arxiv_id=result.get_short_id(),
            title=" ".join(result.title.split()),
            authors=[author.name for author in result.authors],
            abstract=" ".join(result.summary.split()),
            pdf_url=result.pdf_url,
            abs_url=result.entry_id,
            categories=list(result.categories),
            primary_category=result.primary_category,
            published=result.published,
            updated=result.updated,
        )
