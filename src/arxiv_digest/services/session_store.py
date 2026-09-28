"""Local JSON persistence for a paper briefing and its QA conversation history."""

from pathlib import Path

from pydantic import BaseModel, Field

from arxiv_digest.models import ChatTurn, ExecutiveBriefing, PaperMetadata, ParseQuality


class SessionRecord(BaseModel):
    session_id: str
    selected_paper: PaperMetadata
    collection_name: str | None = None
    parse_quality: ParseQuality | None = None
    briefing: ExecutiveBriefing
    chat_history: list[ChatTurn] = Field(default_factory=list)


def session_path(sessions_dir: Path, session_id: str) -> Path:
    return sessions_dir / session_id / "session.json"


def save_session(sessions_dir: Path, session: SessionRecord) -> Path:
    path = session_path(sessions_dir, session.session_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(session.model_dump_json(indent=2), encoding="utf-8")
    return path


def load_session(sessions_dir: Path, session_id: str) -> SessionRecord:
    path = session_path(sessions_dir, session_id)
    if not path.exists():
        raise FileNotFoundError(f"No saved session named '{session_id}' was found.")
    return SessionRecord.model_validate_json(path.read_text(encoding="utf-8"))
