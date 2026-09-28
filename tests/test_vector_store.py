from pathlib import Path

from arxiv_digest.models import TextChunk
from arxiv_digest.services.vector_store import VectorStore, collection_name_for_paper


class FakeEmbedder:
    def encode(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0] if "cache" in text.lower() else [0.0, 1.0] for text in texts]


def chunk(chunk_id: str, text: str, page_number: int) -> TextChunk:
    return TextChunk(
        chunk_id=chunk_id,
        paper_id="2401.12345v1",
        page_number=page_number,
        section="Method",
        text=text,
        word_count=len(text.split()),
    )


def test_vector_store_indexes_and_retrieves_page_citable_chunks(tmp_path: Path) -> None:
    store = VectorStore(tmp_path / "chroma", FakeEmbedder())
    collection_name = collection_name_for_paper("2401.12345v1")
    indexed_count = store.index_chunks(
        collection_name,
        [
            chunk("chunk-1", "KV cache compression result", 4),
            chunk("chunk-2", "dataset details", 8),
        ],
    )

    evidence = store.search(collection_name, "How is the cache compressed?", top_k=1)

    assert indexed_count == 2
    assert evidence[0].chunk_id == "chunk-1"
    assert evidence[0].page_number == 4
    assert evidence[0].distance == 0.0


def test_collection_name_is_stable_and_safe() -> None:
    assert collection_name_for_paper("cs/0112017v3") == "paper_cs_0112017v3"


def test_reindexing_replaces_stale_chunks(tmp_path: Path) -> None:
    store = VectorStore(tmp_path / "chroma", FakeEmbedder())
    collection_name = collection_name_for_paper("2401.12345v1")

    store.index_chunks(
        collection_name,
        [chunk("old-1", "cache details", 1), chunk("old-2", "old", 2)],
    )
    indexed_count = store.index_chunks(collection_name, [chunk("new-1", "cache result", 3)])

    assert indexed_count == 1
