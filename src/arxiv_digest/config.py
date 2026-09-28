"""Central, environment-based configuration for the application."""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """Runtime settings with safe local defaults.

    Values can be overridden in a local .env file or with environment variables
    prefixed with ARXIV_DIGEST_.
    """

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_prefix="ARXIV_DIGEST_",
        extra="ignore",
    )

    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen2.5:3b"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    arxiv_candidate_limit: int = 10
    max_pdf_size_mb: int = 50
    min_extracted_characters: int = 800
    embedding_batch_size: int = 32
    top_k: int = 4
    max_retrieval_distance: float = 0.8
    chunk_size_words: int = 350
    chunk_overlap_words: int = 60

    @property
    def data_dir(self) -> Path:
        return PROJECT_ROOT / "data"

    @property
    def sessions_dir(self) -> Path:
        return self.data_dir / "sessions"

    @property
    def chroma_dir(self) -> Path:
        return self.data_dir / "chroma"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return one shared settings instance for the current process."""

    return Settings()
