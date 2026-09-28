from arxiv_digest.config import PROJECT_ROOT, get_settings


def test_default_settings_use_project_data_directories() -> None:
    settings = get_settings()

    assert settings.data_dir == PROJECT_ROOT / "data"
    assert settings.sessions_dir == PROJECT_ROOT / "data" / "sessions"
    assert settings.chroma_dir == PROJECT_ROOT / "data" / "chroma"
