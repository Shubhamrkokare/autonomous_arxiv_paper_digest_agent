"""Explicit LangGraph workflow coordinating retrieval, processing, briefing, and QA."""

from dataclasses import dataclass
from typing import Literal, TypedDict

from langgraph.graph import END, START, StateGraph

from arxiv_digest.config import Settings
from arxiv_digest.models import (
    ChatTurn,
    ExecutiveBriefing,
    InputKind,
    PaperMetadata,
    ParseQuality,
    ParseResult,
    QAResponse,
    QueryInput,
    RankedPaper,
    RetrievedChunk,
    TextChunk,
)
from arxiv_digest.services.arxiv_client import ArxivService
from arxiv_digest.services.briefing import BriefingService
from arxiv_digest.services.chunker import chunk_pages, save_chunks
from arxiv_digest.services.input_classifier import classify_input
from arxiv_digest.services.llm_client import OllamaClient
from arxiv_digest.services.pdf_processing import (
    PdfProcessingError,
    download_pdf,
    extract_pages,
    save_parse_result,
)
from arxiv_digest.services.ranker import rank_candidates
from arxiv_digest.services.vector_store import (
    SentenceTransformerEmbedder,
    VectorStore,
    collection_name_for_paper,
)


class AgentState(TypedDict, total=False):
    """Shared state passed across every graph node and retained for QA calls."""

    mode: Literal["digest", "qa"]
    user_input: str
    query_input: QueryInput
    candidates: list[PaperMetadata]
    ranked_candidates: list[RankedPaper]
    selected_paper: PaperMetadata
    session_dir: str
    parse_result: ParseResult
    chunks: list[TextChunk]
    collection_name: str
    briefing: ExecutiveBriefing
    question: str
    retrieved_evidence: list[RetrievedChunk]
    qa_response: QAResponse
    chat_history: list[ChatTurn]
    warnings: list[str]
    error: str


@dataclass(frozen=True)
class WorkflowDependencies:
    """Concrete services injected into graph nodes for production or test use."""

    settings: Settings
    arxiv_service: ArxivService
    vector_store: VectorStore
    briefing_service: BriefingService


def _session_directory(settings: Settings, paper: PaperMetadata) -> str:
    """Use one safe, stable local artifact directory for each versioned paper."""

    return str(settings.sessions_dir / collection_name_for_paper(paper.arxiv_id))


def build_workflow(deps: WorkflowDependencies):
    """Build the explicit state graph required by the assessment."""

    def classify_node(state: AgentState) -> dict:
        return {"query_input": classify_input(state["user_input"]), "warnings": []}

    def lookup_node(state: AgentState) -> dict:
        query = state["query_input"]
        paper = deps.arxiv_service.lookup_paper(query.arxiv_id or "")
        return {"selected_paper": paper, "session_dir": _session_directory(deps.settings, paper)}

    def search_node(state: AgentState) -> dict:
        query = state["query_input"]
        candidates = deps.arxiv_service.search_topic(
            query.topic or "", limit=deps.settings.arxiv_candidate_limit
        )
        return {"candidates": candidates}

    def rank_node(state: AgentState) -> dict:
        ranked = rank_candidates(state["query_input"].topic or "", state.get("candidates", []))
        if not ranked:
            return {"error": "No arXiv papers matched this topic."}
        return {"ranked_candidates": ranked}

    def select_node(state: AgentState) -> dict:
        paper = state["ranked_candidates"][0].paper
        return {"selected_paper": paper, "session_dir": _session_directory(deps.settings, paper)}

    def parse_node(state: AgentState) -> dict:
        paper = state["selected_paper"]
        session_dir = deps.settings.sessions_dir / collection_name_for_paper(paper.arxiv_id)
        try:
            pdf_path = download_pdf(
                paper.pdf_url,
                session_dir / "paper.pdf",
                max_size_bytes=deps.settings.max_pdf_size_mb * 1024 * 1024,
            )
            result = extract_pages(pdf_path, min_characters=deps.settings.min_extracted_characters)
            save_parse_result(result, session_dir / "pages.json")
            return {"parse_result": result, "warnings": result.warnings}
        except PdfProcessingError as error:
            return {"error": str(error)}

    def index_node(state: AgentState) -> dict:
        parsed = state["parse_result"]
        paper = state["selected_paper"]
        chunks = chunk_pages(
            parsed.pages,
            paper_id=paper.arxiv_id,
            chunk_size_words=deps.settings.chunk_size_words,
            chunk_overlap_words=deps.settings.chunk_overlap_words,
        )
        session_dir = deps.settings.sessions_dir / collection_name_for_paper(paper.arxiv_id)
        save_chunks(chunks, session_dir / "chunks.json")
        collection_name = collection_name_for_paper(paper.arxiv_id)
        deps.vector_store.index_chunks(collection_name, chunks)
        return {"chunks": chunks, "collection_name": collection_name}

    def briefing_node(state: AgentState) -> dict:
        parsed = state.get("parse_result")
        briefing = deps.briefing_service.generate_briefing(
            state["selected_paper"],
            state.get("chunks", []),
            abstract_only=parsed is None or parsed.quality is ParseQuality.ABSTRACT_ONLY,
        )
        return {"briefing": briefing}

    def qa_node(state: AgentState) -> dict:
        evidence = deps.vector_store.search(
            state["collection_name"], state["question"], top_k=deps.settings.top_k
        )
        response = deps.briefing_service.answer_question(
            state["question"], evidence, max_distance=deps.settings.max_retrieval_distance
        )
        history = list(state.get("chat_history", []))
        history.append(ChatTurn(question=state["question"], response=response))
        return {"retrieved_evidence": evidence, "qa_response": response, "chat_history": history}

    def route_initial(state: AgentState) -> str:
        return "answer_question" if state.get("mode") == "qa" else "classify_input"

    def route_input(state: AgentState) -> str:
        if state["query_input"].kind is InputKind.ARXIV_ID:
            return "lookup_paper"
        return "search_papers"

    def route_rank(state: AgentState) -> str:
        return "failed" if state.get("error") else "select_paper"

    def route_parse(state: AgentState) -> str:
        if state.get("error"):
            return "failed"
        if state["parse_result"].quality is ParseQuality.FULL_TEXT:
            return "chunk_and_index"
        return "generate_briefing"

    workflow = StateGraph(AgentState)
    workflow.add_node("classify_input", classify_node)
    workflow.add_node("lookup_paper", lookup_node)
    workflow.add_node("search_papers", search_node)
    workflow.add_node("rank_candidates", rank_node)
    workflow.add_node("select_paper", select_node)
    workflow.add_node("fetch_and_parse", parse_node)
    workflow.add_node("chunk_and_index", index_node)
    workflow.add_node("generate_briefing", briefing_node)
    workflow.add_node("answer_question", qa_node)
    workflow.add_node("failed", lambda state: state)
    workflow.add_conditional_edges(START, route_initial)
    workflow.add_conditional_edges("classify_input", route_input)
    workflow.add_edge("lookup_paper", "fetch_and_parse")
    workflow.add_edge("search_papers", "rank_candidates")
    workflow.add_conditional_edges("rank_candidates", route_rank)
    workflow.add_edge("select_paper", "fetch_and_parse")
    workflow.add_conditional_edges("fetch_and_parse", route_parse)
    workflow.add_edge("chunk_and_index", "generate_briefing")
    workflow.add_edge("generate_briefing", END)
    workflow.add_edge("answer_question", END)
    workflow.add_edge("failed", END)
    return workflow.compile()


def build_production_workflow() -> tuple[object, Settings]:
    """Assemble the local-only production dependencies used by the CLI."""

    from arxiv_digest.config import get_settings

    settings = get_settings()
    vector_store = VectorStore(
        settings.chroma_dir,
        SentenceTransformerEmbedder(settings.embedding_model),
    )
    briefing_service = BriefingService(
        OllamaClient(settings.ollama_base_url, settings.ollama_model)
    )
    workflow = build_workflow(
        WorkflowDependencies(
            settings=settings,
            arxiv_service=ArxivService(),
            vector_store=vector_store,
            briefing_service=briefing_service,
        )
    )
    return workflow, settings
