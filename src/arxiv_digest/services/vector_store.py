"""Local embedding and ChromaDB storage used for grounded paper QA."""

import re
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

import chromadb

from arxiv_digest.models import RetrievedChunk, TextChunk


class Embedder(Protocol):
    """Minimal interface that lets tests avoid loading a real ML model."""

    def encode(self, texts: Sequence[str]) -> list[list[float]]: ...


class SentenceTransformerEmbedder:
    """Lazy wrapper around a local sentence-transformers embedding model."""

    def __init__(self, model_name: str) -> None:
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name)

    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        vectors = self._model.encode(
            list(texts),
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return vectors.tolist()


def collection_name_for_paper(paper_id: str) -> str:
    """Create a stable Chroma-safe collection name from a versioned arXiv ID."""

    normalized = re.sub(r"[^a-z0-9_-]+", "_", paper_id.lower())
    return f"paper_{normalized}"


class VectorStore:
    """Persist chunks and perform semantic search without a hosted vector database."""

    def __init__(self, directory: Path, embedder: Embedder) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(directory))
        self._embedder = embedder

    def index_chunks(self, collection_name: str, chunks: list[TextChunk]) -> int:
        """Replace a paper's chunks, so changing chunk settings cannot leave stale vectors."""

        if not chunks:
            return 0

        collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        paper_ids = {chunk.paper_id for chunk in chunks}
        if len(paper_ids) != 1:
            raise ValueError("A Chroma collection must contain chunks from exactly one paper.")
        collection.delete(where={"paper_id": next(iter(paper_ids))})
        collection.upsert(
            ids=[chunk.chunk_id for chunk in chunks],
            documents=[chunk.text for chunk in chunks],
            embeddings=self._embedder.encode([chunk.text for chunk in chunks]),
            metadatas=[
                {
                    "paper_id": chunk.paper_id,
                    "page_number": chunk.page_number,
                    "section": chunk.section,
                    "word_count": chunk.word_count,
                }
                for chunk in chunks
            ],
        )
        return collection.count()

    def search(self, collection_name: str, question: str, *, top_k: int) -> list[RetrievedChunk]:
        """Return the most similar evidence chunks and their page-citation metadata."""

        collection = self._client.get_collection(collection_name)
        available_chunks = collection.count()
        if available_chunks == 0:
            return []

        results = collection.query(
            query_embeddings=self._embedder.encode([question]),
            n_results=min(top_k, available_chunks),
            include=["documents", "metadatas", "distances"],
        )
        ids = results["ids"][0]
        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]

        return [
            RetrievedChunk(
                chunk_id=chunk_id,
                paper_id=str(metadata["paper_id"]),
                page_number=int(metadata["page_number"]),
                section=str(metadata["section"]),
                text=str(document),
                word_count=int(metadata["word_count"]),
                distance=float(distance),
            )
            for chunk_id, document, metadata, distance in zip(ids, documents, metadatas, distances)
        ]
