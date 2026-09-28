from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

from arxiv_digest.services.arxiv_client import ArxivRetrievalError, ArxivService, build_topic_query


class FakeArxivClient:
    def __init__(self, results: list[object]) -> None:
        self.results_to_return = results
        self.searches: list[object] = []

    def results(self, search: object) -> list[object]:
        self.searches.append(search)
        return self.results_to_return


def make_result() -> object:
    return SimpleNamespace(
        get_short_id=lambda: "2401.12345v2",
        title="A   Paper\nWith  Spacing",
        authors=[SimpleNamespace(name="Ada Lovelace"), SimpleNamespace(name="Grace Hopper")],
        summary="An  abstract\nwith extra spacing.",
        pdf_url="https://arxiv.org/pdf/2401.12345v2",
        entry_id="https://arxiv.org/abs/2401.12345v2",
        categories=["cs.AI", "cs.CL"],
        primary_category="cs.AI",
        published=datetime(2024, 1, 20, tzinfo=UTC),
        updated=datetime(2024, 1, 22, tzinfo=UTC),
    )


def test_lookup_normalizes_arxiv_result() -> None:
    service = ArxivService(client=FakeArxivClient([make_result()]))

    paper = service.lookup_paper("2401.12345")

    assert paper.arxiv_id == "2401.12345v2"
    assert paper.title == "A Paper With Spacing"
    assert paper.abstract == "An abstract with extra spacing."
    assert paper.authors == ["Ada Lovelace", "Grace Hopper"]
    assert paper.categories == ["cs.AI", "cs.CL"]


def test_lookup_raises_when_arxiv_returns_nothing() -> None:
    service = ArxivService(client=FakeArxivClient([]))

    with pytest.raises(ArxivRetrievalError, match="No arXiv paper"):
        service.lookup_paper("2401.12345")


def test_build_topic_query_uses_explicit_all_fields_and_ignores_filler_words() -> None:
    query = build_topic_query("recent work on KV cache compression for LLMs")

    assert query == "all:kv AND all:cache AND all:compression AND all:llms"


def test_topic_search_uses_a_fielded_relevance_query() -> None:
    client = FakeArxivClient([make_result()])
    service = ArxivService(client=client)

    papers = service.search_topic("KV cache", limit=5)

    assert len(papers) == 1
    assert client.searches[0].query == "all:kv AND all:cache"
    assert client.searches[0].sort_by.name == "Relevance"
