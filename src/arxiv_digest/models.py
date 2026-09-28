"""Typed domain models shared by graph nodes and services."""

from datetime import datetime
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, Field, model_validator


class InputKind(StrEnum):
    """The two input modes accepted by the assessment."""

    ARXIV_ID = "arxiv_id"
    TOPIC = "topic"


class QueryInput(BaseModel):
    """Normalized result of classifying raw user input."""

    raw_input: str
    kind: InputKind
    arxiv_id: str | None = None
    topic: str | None = None

    @model_validator(mode="after")
    def validate_mode_fields(self) -> "QueryInput":
        """Ensure the normalized form always has exactly one active input mode."""

        if self.kind is InputKind.ARXIV_ID and (not self.arxiv_id or self.topic is not None):
            raise ValueError("An arXiv-ID input must contain only arxiv_id.")
        if self.kind is InputKind.TOPIC and (not self.topic or self.arxiv_id is not None):
            raise ValueError("A topic input must contain only topic.")
        return self


class PaperMetadata(BaseModel):
    """The arXiv fields required by the briefing and later retrieval stages."""

    arxiv_id: str
    title: str
    authors: list[str]
    abstract: str
    pdf_url: str
    abs_url: str
    categories: list[str] = Field(default_factory=list)
    primary_category: str | None = None
    published: datetime
    updated: datetime | None = None


class RankedPaper(BaseModel):
    """A topic-search candidate plus explainable deterministic ranking details."""

    paper: PaperMetadata
    relevance_score: float
    matched_terms: list[str] = Field(default_factory=list)


class ParseQuality(StrEnum):
    """Whether a PDF supplied enough extracted text for full-paper QA."""

    FULL_TEXT = "full_text"
    ABSTRACT_ONLY = "abstract_only"


class ParsedPage(BaseModel):
    """Text extracted from one PDF page, retaining its human-readable location."""

    page_number: int = Field(ge=1)
    text: str


class ParseResult(BaseModel):
    """Page-aware parsing outcome used by chunking and fallback logic."""

    pdf_path: Path
    quality: ParseQuality
    pages: list[ParsedPage] = Field(default_factory=list)
    total_characters: int = Field(ge=0)
    warnings: list[str] = Field(default_factory=list)


class TextChunk(BaseModel):
    """A searchable passage with the minimum metadata needed for citations."""

    chunk_id: str
    paper_id: str
    page_number: int = Field(ge=1)
    section: str
    text: str = Field(min_length=1)
    word_count: int = Field(ge=1)


class RetrievedChunk(TextChunk):
    """A chunk returned by similarity search, including its distance score."""

    distance: float = Field(ge=0)


class EvidenceCitation(BaseModel):
    """A verifiable pointer to a chunk supplied to the answering model."""

    chunk_id: str
    page_number: int = Field(ge=1)
    section: str


class ExecutiveBriefing(BaseModel):
    """The structured artifact explicitly required by the assessment."""

    title: str
    authors: list[str]
    arxiv_id: str
    publish_date: str
    link: str
    why_it_matters: str
    problem_statement: str
    method_approach: list[str]
    key_results_claims: list[str]
    limitations: list[str]
    suggested_follow_up_questions: list[str]


class QAResponse(BaseModel):
    """A grounded answer or an explicit statement that evidence was insufficient."""

    answerable: bool
    answer: str
    citations: list[EvidenceCitation] = Field(default_factory=list)


class ChatTurn(BaseModel):
    """Persisted record of one user question and the agent's response."""

    question: str
    response: QAResponse
