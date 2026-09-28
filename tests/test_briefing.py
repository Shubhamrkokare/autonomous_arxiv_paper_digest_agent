from datetime import UTC, datetime
from typing import Any

from arxiv_digest.models import PaperMetadata, RetrievedChunk, TextChunk
from arxiv_digest.services.briefing import INSUFFICIENT_EVIDENCE_MESSAGE, BriefingService


class FakeLLM:
    def __init__(self, responses: list[dict[str, Any]]) -> None:
        self.responses = responses
        self.prompts: list[str] = []

    def generate_json(self, prompt: str, *, schema: dict[str, Any] | None = None) -> dict[str, Any]:
        self.prompts.append(prompt)
        return self.responses.pop(0)


def paper() -> PaperMetadata:
    return PaperMetadata(
        arxiv_id="2401.12345v1",
        title="Paper title",
        authors=["Ada Lovelace"],
        abstract="Paper abstract",
        pdf_url="https://arxiv.org/pdf/2401.12345v1",
        abs_url="https://arxiv.org/abs/2401.12345v1",
        published=datetime(2024, 1, 20, tzinfo=UTC),
    )


def test_briefing_overwrites_identity_fields_with_trusted_metadata() -> None:
    llm = FakeLLM(
        [
            {
                "title": "Wrong title",
                "authors": [],
                "arxiv_id": "wrong",
                "publish_date": "wrong",
                "link": "wrong",
                "why_it_matters": "Useful",
                "problem_statement": "Problem",
                "method_approach": ["Method"],
                "key_results_claims": ["Result"],
                "limitations": ["Not stated"],
                "suggested_follow_up_questions": ["Why?"],
            }
        ]
    )
    service = BriefingService(llm)
    chunks = [
        TextChunk(
            chunk_id="chunk-1",
            paper_id="2401.12345v1",
            page_number=2,
            section="Method",
            text="Evidence text",
            word_count=2,
        )
    ]

    briefing = service.generate_briefing(paper(), chunks)

    assert briefing.title == "Paper title"
    assert briefing.authors == ["Ada Lovelace"]
    assert briefing.arxiv_id == "2401.12345v1"
    assert "Evidence text" in llm.prompts[0]


def test_briefing_extracts_only_author_caveats_for_limitations() -> None:
    llm = FakeLLM(
        [
            {
                "why_it_matters": "Useful",
                "problem_statement": "Problem",
                "method_approach": ["Method"],
                "key_results_claims": ["Result"],
                "suggested_follow_up_questions": ["Why?"],
            }
        ]
    )
    chunks = [
        TextChunk(
            chunk_id="chunk-1",
            paper_id="2401.12345v1",
            page_number=3,
            section="Discussion",
            text="The authors plan future work to address the restricted setting.",
            word_count=10,
        )
    ]

    briefing = BriefingService(llm).generate_briefing(paper(), chunks)

    assert briefing.limitations == [
        "Author-stated caveat (p. 3): "
        "The authors plan future work to address the restricted setting."
    ]


def test_briefing_limits_generated_text_fields_to_two_sentences() -> None:
    llm = FakeLLM(
        [
            {
                "why_it_matters": "First. Second. Third.",
                "problem_statement": "One. Two. Three.",
                "method_approach": ["Method"],
                "key_results_claims": ["Result"],
                "suggested_follow_up_questions": ["Why?"],
            }
        ]
    )

    briefing = BriefingService(llm).generate_briefing(paper(), [])

    assert briefing.why_it_matters == "First. Second."
    assert briefing.problem_statement == "One. Two."


def test_qa_refuses_weak_evidence_without_calling_the_llm() -> None:
    llm = FakeLLM([])
    service = BriefingService(llm)
    evidence = [
        RetrievedChunk(
            chunk_id="chunk-1",
            paper_id="paper",
            page_number=1,
            section="Method",
            text="Irrelevant text",
            word_count=2,
            distance=0.95,
        )
    ]

    response = service.answer_question("What is the result?", evidence, max_distance=0.8)

    assert response.answerable is False
    assert response.answer == INSUFFICIENT_EVIDENCE_MESSAGE
    assert llm.prompts == []


def test_qa_keeps_only_citations_that_match_retrieved_evidence() -> None:
    llm = FakeLLM(
        [
            {
                "answerable": True,
                "answer": "The paper reports a result.",
                "citations": [
                    {"chunk_id": "chunk-1", "page_number": 999, "section": "Invented"},
                    {"chunk_id": "not-retrieved", "page_number": 2, "section": "No"},
                ],
            }
        ]
    )
    evidence = [
        RetrievedChunk(
            chunk_id="chunk-1",
            paper_id="paper",
            page_number=4,
            section="Results",
            text="The result is 90%.",
            word_count=4,
            distance=0.2,
        )
    ]

    response = BriefingService(llm).answer_question(
        "What is the result?", evidence, max_distance=0.8
    )

    assert response.answerable is True
    assert response.citations[0].page_number == 4
    assert response.citations[0].section == "Results"


def test_qa_normalizes_model_refusals_to_the_safe_message() -> None:
    llm = FakeLLM(
        [
            {
                "answerable": False,
                "answer": "The model's longer refusal must not be exposed.",
                "citations": [],
            }
        ]
    )
    evidence = [
        RetrievedChunk(
            chunk_id="chunk-1",
            paper_id="paper",
            page_number=1,
            section="Introduction",
            text="Some paper evidence.",
            word_count=3,
            distance=0.2,
        )
    ]

    response = BriefingService(llm).answer_question(
        "What is the office address?", evidence, max_distance=0.8
    )

    assert response.answerable is False
    assert response.answer == INSUFFICIENT_EVIDENCE_MESSAGE
    assert response.citations == []


def test_qa_refuses_an_answer_that_has_only_invented_citations() -> None:
    llm = FakeLLM(
        [
            {
                "answerable": True,
                "answer": "Unsupported claim.",
                "citations": [{"chunk_id": "invented", "page_number": 99, "section": "Fake"}],
            }
        ]
    )
    evidence = [
        RetrievedChunk(
            chunk_id="chunk-1",
            paper_id="paper",
            page_number=1,
            section="Introduction",
            text="Some paper evidence.",
            word_count=3,
            distance=0.2,
        )
    ]

    response = BriefingService(llm).answer_question("Unsupported?", evidence, max_distance=0.8)

    assert response.answerable is False
    assert response.answer == INSUFFICIENT_EVIDENCE_MESSAGE
