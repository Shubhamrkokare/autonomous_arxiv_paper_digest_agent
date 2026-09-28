from datetime import UTC, datetime

from arxiv_digest.models import PaperMetadata
from arxiv_digest.services.ranker import rank_candidates


def paper(arxiv_id: str, title: str, abstract: str, year: int) -> PaperMetadata:
    return PaperMetadata(
        arxiv_id=arxiv_id,
        title=title,
        authors=["Test Author"],
        abstract=abstract,
        pdf_url=f"https://arxiv.org/pdf/{arxiv_id}",
        abs_url=f"https://arxiv.org/abs/{arxiv_id}",
        published=datetime(year, 1, 1, tzinfo=UTC),
    )


def test_ranker_weights_title_matches_more_than_abstract_matches() -> None:
    title_match = paper("1", "KV cache compression", "A systems paper", 2024)
    abstract_match = paper("2", "Efficient systems", "We study KV cache compression.", 2025)

    results = rank_candidates("recent KV cache compression work", [abstract_match, title_match])

    assert [item.paper.arxiv_id for item in results] == ["1", "2"]
    assert results[0].matched_terms == ["cache", "compression", "kv"]
    assert results[0].relevance_score == 9.0


def test_ranker_uses_newer_papers_to_break_equal_scores() -> None:
    older = paper("1", "Unrelated title", "Unrelated abstract", 2023)
    newer = paper("2", "Another title", "Another abstract", 2025)

    results = rank_candidates("KV cache", [older, newer])

    assert [item.paper.arxiv_id for item in results] == ["2", "1"]
