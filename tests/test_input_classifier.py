import pytest

from arxiv_digest.models import InputKind, QueryInput
from arxiv_digest.services.input_classifier import InputValidationError, classify_input


@pytest.mark.parametrize(
    ("raw_input", "expected_id"),
    [
        ("2401.12345", "2401.12345"),
        ("arXiv:2401.12345v2", "2401.12345v2"),
        ("cs/0112017", "cs/0112017"),
        ("https://arxiv.org/abs/2401.12345", "2401.12345"),
        ("https://arxiv.org/pdf/2401.12345v2.pdf", "2401.12345v2"),
    ],
)
def test_classify_arxiv_identifiers(raw_input: str, expected_id: str) -> None:
    result = classify_input(raw_input)

    assert result.kind is InputKind.ARXIV_ID
    assert result.arxiv_id == expected_id
    assert result.topic is None


def test_classify_topic() -> None:
    result = classify_input("recent work on KV-cache compression for LLMs")

    assert result.kind is InputKind.TOPIC
    assert result.topic == "recent work on KV-cache compression for LLMs"
    assert result.arxiv_id is None


@pytest.mark.parametrize("raw_input", ["", "   ", "https://example.com/paper"])
def test_reject_empty_or_unsupported_url(raw_input: str) -> None:
    with pytest.raises(InputValidationError):
        classify_input(raw_input)


def test_query_input_rejects_mixed_modes() -> None:
    with pytest.raises(ValueError, match="only arxiv_id"):
        QueryInput(
            raw_input="2401.12345",
            kind=InputKind.ARXIV_ID,
            arxiv_id="2401.12345",
            topic="a topic that should not coexist",
        )
