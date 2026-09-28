from arxiv_digest.config import get_settings
from arxiv_digest.graph import WorkflowDependencies, build_workflow
from arxiv_digest.services.arxiv_client import ArxivService
from arxiv_digest.services.briefing import BriefingService
from arxiv_digest.services.vector_store import VectorStore


class FakeEmbedder:
    def encode(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0] for _ in texts]


class FakeLLM:
    def generate_json(self, prompt: str, *, schema: dict | None = None) -> dict:
        return {}


def test_graph_contains_explicit_digest_and_qa_nodes(tmp_path) -> None:
    dependencies = WorkflowDependencies(
        settings=get_settings(),
        arxiv_service=ArxivService(client=object()),
        vector_store=VectorStore(tmp_path / "chroma", FakeEmbedder()),
        briefing_service=BriefingService(FakeLLM()),
    )

    workflow = build_workflow(dependencies)

    expected_nodes = {
        "classify_input",
        "fetch_and_parse",
        "chunk_and_index",
        "generate_briefing",
        "answer_question",
    }
    assert expected_nodes <= set(workflow.get_graph().nodes)
