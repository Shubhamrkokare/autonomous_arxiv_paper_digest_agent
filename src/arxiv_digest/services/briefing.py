"""Grounded executive-briefing and QA generation from supplied paper evidence."""

import json
import re

from pydantic import BaseModel

from arxiv_digest.models import (
    EvidenceCitation,
    ExecutiveBriefing,
    PaperMetadata,
    QAResponse,
    RetrievedChunk,
    TextChunk,
)
from arxiv_digest.services.llm_client import LLMClient

INSUFFICIENT_EVIDENCE_MESSAGE = (
    "I could not find sufficient evidence in the indexed content of this paper "
    "to answer that question."
)


class BriefingAnalysis(BaseModel):
    """Only fields the model is allowed to author from supplied evidence."""

    why_it_matters: str
    problem_statement: str
    method_approach: list[str]
    key_results_claims: list[str]
    suggested_follow_up_questions: list[str]


class BriefingService:
    """Uses an LLM only to phrase claims that originate in selected paper text."""

    def __init__(self, llm: LLMClient, *, max_evidence_characters: int = 24_000) -> None:
        self._llm = llm
        self._max_evidence_characters = max_evidence_characters

    def generate_briefing(
        self, paper: PaperMetadata, chunks: list[TextChunk], *, abstract_only: bool = False
    ) -> ExecutiveBriefing:
        """Create the required briefing schema from metadata and bounded paper evidence."""

        evidence = self._briefing_evidence(chunks) or paper.abstract
        availability_note = (
            "Only the abstract is available; label uncertainty clearly." if abstract_only else ""
        )
        prompt = (
            "You are preparing a factual executive briefing for one arXiv paper. "
            "Use only the metadata and paper evidence below. Do not invent details. "
            "Write concisely: at most two sentences for text fields and at most three bullets for "
            "each list. Do not speculate about limitations: this system extracts author-stated "
            "caveats directly from the paper instead. "
            f"{availability_note}\n"
            "Return JSON only. Provide every field in the requested schema, and use short "
            "evidence-based bullets in list fields.\n\n"
            f"PAPER EVIDENCE:\n{evidence}\n"
        )
        analysis = BriefingAnalysis.model_validate(
            self._llm.generate_json(prompt, schema=BriefingAnalysis.model_json_schema())
        )
        analysis_data = analysis.model_dump()
        for field_name in (
            "method_approach",
            "key_results_claims",
            "suggested_follow_up_questions",
        ):
            analysis_data[field_name] = analysis_data[field_name][:3]
        return ExecutiveBriefing(
            title=paper.title,
            authors=paper.authors,
            arxiv_id=paper.arxiv_id,
            publish_date=paper.published.date().isoformat(),
            link=paper.abs_url,
            limitations=self._author_stated_limitations(chunks),
            **analysis_data,
        )

    def answer_question(
        self,
        question: str,
        evidence: list[RetrievedChunk],
        *,
        max_distance: float,
    ) -> QAResponse:
        """Answer only when retrieval passed an application-controlled evidence gate."""

        if not evidence or evidence[0].distance > max_distance:
            return QAResponse(answerable=False, answer=INSUFFICIENT_EVIDENCE_MESSAGE)

        allowed_citations = {
            chunk.chunk_id: EvidenceCitation(
                chunk_id=chunk.chunk_id,
                page_number=chunk.page_number,
                section=chunk.section,
            )
            for chunk in evidence
        }
        prompt = (
            "Answer the user's question using only the supplied paper evidence. "
            "If the evidence does not support an answer, set answerable to false and say so. "
            "Return JSON only with keys answerable, answer, citations. Each citation must use one "
            "of the exact chunk_id values in the evidence. Do not cite anything else.\n\n"
            f"QUESTION: {question}\n\n"
            f"EVIDENCE:\n{json.dumps([chunk.model_dump() for chunk in evidence], default=str)}\n"
        )
        response = QAResponse.model_validate(
            self._llm.generate_json(prompt, schema=QAResponse.model_json_schema())
        )
        if not response.answerable:
            return QAResponse(answerable=False, answer=INSUFFICIENT_EVIDENCE_MESSAGE)
        verified_citations = [
            allowed_citations[citation.chunk_id]
            for citation in response.citations
            if citation.chunk_id in allowed_citations
        ]
        if response.answerable and not verified_citations:
            return QAResponse(answerable=False, answer=INSUFFICIENT_EVIDENCE_MESSAGE)
        return response.model_copy(update={"citations": verified_citations})

    def _briefing_evidence(self, chunks: list[TextChunk]) -> str:
        selected: list[str] = []
        used_characters = 0
        for chunk in chunks:
            source = f"[page {chunk.page_number}; {chunk.section}]\n{chunk.text}\n"
            if used_characters + len(source) > self._max_evidence_characters:
                break
            selected.append(source)
            used_characters += len(source)
        return "\n".join(selected)

    @staticmethod
    def _author_stated_limitations(chunks: list[TextChunk]) -> list[str]:
        """Return verbatim author caveats instead of model-inferred limitations.

        The assessment asks for limitations stated by the authors. A lightweight,
        deterministic extraction makes the boundary auditable and avoids treating
        arbitrary missing details as a limitation.
        """

        direct_cues = ("future work", "we plan", "cannot", "unable")
        own_work_cues = ("our ", "this approach", "this method", "the proposed")
        limitations: list[str] = []
        for chunk in chunks:
            sentences = re.split(r"(?<=[.!?])\s+", chunk.text.replace("\n", " "))
            for sentence in sentences:
                normalized = sentence.strip()
                lowered = normalized.lower()
                is_direct_caveat = any(cue in lowered for cue in direct_cues)
                is_own_limitation = "limitation" in lowered and any(
                    cue in lowered for cue in own_work_cues
                )
                if len(normalized) < 30 or not (is_direct_caveat or is_own_limitation):
                    continue
                limitations.append(f"Author-stated caveat (p. {chunk.page_number}): {normalized}")
                if len(limitations) == 3:
                    return limitations
        return limitations or [
            "The paper does not explicitly state limitations in the supplied evidence."
        ]
