from datetime import UTC, datetime

import pytest

from arxiv_digest.models import ExecutiveBriefing, PaperMetadata
from arxiv_digest.services.session_store import SessionRecord, load_session, save_session


def make_session() -> SessionRecord:
    paper = PaperMetadata(
        arxiv_id="2401.12345v1",
        title="Paper title",
        authors=["Ada Lovelace"],
        abstract="Abstract",
        pdf_url="https://arxiv.org/pdf/2401.12345v1",
        abs_url="https://arxiv.org/abs/2401.12345v1",
        published=datetime(2024, 1, 20, tzinfo=UTC),
    )
    briefing = ExecutiveBriefing(
        title=paper.title,
        authors=paper.authors,
        arxiv_id=paper.arxiv_id,
        publish_date="2024-01-20",
        link=paper.abs_url,
        why_it_matters="Why",
        problem_statement="Problem",
        method_approach=["Method"],
        key_results_claims=["Result"],
        limitations=["Limit"],
        suggested_follow_up_questions=["Question"],
    )
    return SessionRecord(
        session_id="paper_2401_12345v1",
        selected_paper=paper,
        collection_name="paper_2401_12345v1",
        briefing=briefing,
    )


def test_session_round_trip(tmp_path) -> None:
    session = make_session()

    path = save_session(tmp_path, session)
    restored = load_session(tmp_path, session.session_id)

    assert path.exists()
    assert restored == session


def test_loading_missing_session_raises_actionable_error(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="No saved session"):
        load_session(tmp_path, "missing")
