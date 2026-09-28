"""Explainable, dependency-free ranking for topic-search candidates."""

import re

from arxiv_digest.models import PaperMetadata, RankedPaper

TOKEN_PATTERN = re.compile(r"[a-z0-9]+")
STOP_WORDS = {
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


def _meaningful_terms(text: str) -> set[str]:
    """Extract simple lowercase query terms without pretending to understand language."""

    return {
        token
        for token in TOKEN_PATTERN.findall(text.lower())
        if token not in STOP_WORDS and len(token) >= 2
    }


def rank_candidates(topic: str, candidates: list[PaperMetadata]) -> list[RankedPaper]:
    """Rank candidates by transparent lexical overlap, then publication date.

    Title matches count three times as much as abstract matches. The output keeps
    matched terms so a CLI or future UI can explain the ranking to the user.
    """

    query_terms = _meaningful_terms(topic)
    ranked: list[RankedPaper] = []

    for paper in candidates:
        title_terms = _meaningful_terms(paper.title)
        abstract_terms = _meaningful_terms(paper.abstract)
        title_matches = query_terms & title_terms
        abstract_matches = query_terms & abstract_terms
        matched_terms = sorted(title_matches | abstract_matches)
        score = float(3 * len(title_matches) + len(abstract_matches - title_matches))

        ranked.append(
            RankedPaper(
                paper=paper,
                relevance_score=score,
                matched_terms=matched_terms,
            )
        )

    return sorted(
        ranked,
        key=lambda item: (item.relevance_score, item.paper.published),
        reverse=True,
    )
