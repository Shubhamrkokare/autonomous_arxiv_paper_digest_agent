"""Command-line entry point for the arXiv digest agent."""

import json
from typing import Annotated

import typer

from arxiv_digest import __version__
from arxiv_digest.config import get_settings
from arxiv_digest.graph import build_production_workflow
from arxiv_digest.models import InputKind
from arxiv_digest.services.arxiv_client import ArxivRetrievalError, ArxivService
from arxiv_digest.services.input_classifier import InputValidationError, classify_input
from arxiv_digest.services.llm_client import LLMUnavailableError
from arxiv_digest.services.session_store import SessionRecord, load_session, save_session

app = typer.Typer(
    name="arxiv-digest",
    help="Create grounded executive briefings and QA sessions for arXiv papers.",
    no_args_is_help=True,
)


def version_callback(value: bool) -> None:
    if value:
        typer.echo(__version__)
        raise typer.Exit()


@app.callback()
def root(
    version: bool = typer.Option(
        False,
        "--version",
        callback=version_callback,
        is_eager=True,
        help="Show the installed version and exit.",
    ),
) -> None:
    """Run the autonomous arXiv paper digest and QA agent."""


def main() -> None:
    """Run the Typer application for the installed console-script entry point."""

    app()


@app.command()
def inspect(
    query: Annotated[
        str,
        typer.Argument(help="An arXiv ID, arXiv URL, or natural-language research topic."),
    ],
) -> None:
    """Classify input and retrieve raw arXiv metadata or topic candidates."""

    try:
        classified = classify_input(query)
        service = ArxivService()
        if classified.kind is InputKind.ARXIV_ID:
            papers = [service.lookup_paper(classified.arxiv_id or "")]
        else:
            papers = service.search_topic(
                classified.topic or "", limit=get_settings().arxiv_candidate_limit
            )
    except (InputValidationError, ArxivRetrievalError) as error:
        typer.echo(f"Error: {error}", err=True)
        raise typer.Exit(code=1) from error

    payload = {
        "input": classified.model_dump(mode="json"),
        "paper_count": len(papers),
        "papers": [paper.model_dump(mode="json") for paper in papers],
    }
    typer.echo(json.dumps(payload, indent=2))


@app.command()
def digest(
    query: Annotated[
        str,
        typer.Argument(help="An arXiv ID, arXiv URL, or natural-language research topic."),
    ],
) -> None:
    """Create and save a grounded executive briefing for one selected arXiv paper."""

    try:
        workflow, settings = build_production_workflow()
        result = workflow.invoke({"mode": "digest", "user_input": query})
        if result.get("error"):
            raise RuntimeError(result["error"])
        paper = result["selected_paper"]
        session_id = result["collection_name"] or f"abstract_{paper.arxiv_id.replace('/', '_')}"
        session = SessionRecord(
            session_id=session_id,
            selected_paper=paper,
            collection_name=result.get("collection_name"),
            parse_quality=(
                result.get("parse_result").quality if result.get("parse_result") else None
            ),
            briefing=result["briefing"],
            chat_history=[],
        )
        save_session(settings.sessions_dir, session)
    except (ArxivRetrievalError, InputValidationError, LLMUnavailableError, RuntimeError) as error:
        typer.echo(f"Error: {error}", err=True)
        raise typer.Exit(code=1) from error

    typer.echo(session.briefing.model_dump_json(indent=2))
    typer.echo(f"\nSession saved: {session.session_id}")


@app.command()
def qa(
    session_id: Annotated[str, typer.Option(help="Session ID printed by the digest command.")],
    question: Annotated[str, typer.Argument(help="Question about the selected paper.")],
) -> None:
    """Answer a question using only retrieved chunks from a saved full-text session."""

    try:
        workflow, settings = build_production_workflow()
        session = load_session(settings.sessions_dir, session_id)
        if not session.collection_name:
            raise RuntimeError(
                "This session has no full-text index, so detailed QA is unavailable."
            )
        result = workflow.invoke(
            {
                "mode": "qa",
                "selected_paper": session.selected_paper,
                "collection_name": session.collection_name,
                "question": question,
                "chat_history": session.chat_history,
            }
        )
        session.chat_history = result["chat_history"]
        save_session(settings.sessions_dir, session)
    except (FileNotFoundError, LLMUnavailableError, RuntimeError) as error:
        typer.echo(f"Error: {error}", err=True)
        raise typer.Exit(code=1) from error

    typer.echo(result["qa_response"].model_dump_json(indent=2))


if __name__ == "__main__":
    main()
