# Interview Guide: Autonomous arXiv Paper Digest & QA Agent

This guide explains the project in the order it runs. The central principle is simple: the language model writes prose, but the application controls what information it may use and when it must refuse.

## 1. One-minute architecture explanation

The CLI accepts either an arXiv ID/URL or a topic. An ID takes the exact arXiv lookup path. A topic takes search, deterministic ranking, and selection. The selected paper is downloaded, parsed page by page, chunked, embedded locally, and stored in ChromaDB. A local Ollama model produces a structured briefing. For later questions, the system retrieves only the nearest chunks, checks their cosine distance, and asks the model to answer from those chunks only. Citations are verified in application code before they are returned.

LangGraph makes this flow explicit. `AgentState` is the shared state object and every node returns a small update to it. Conditional edges select the ID/topic route, the full-text/abstract-only route, and the QA/refusal route.

## 2. Why each component exists

| Component | What it does | Why this choice |
|---|---|---|
| `input_classifier.py` | Recognizes arXiv IDs and URLs versus topics | Prevents arbitrary URLs from entering the retrieval pipeline. |
| `arxiv_client.py` | Fetches metadata and topic candidates from arXiv | arXiv is the authoritative source for title, authors, dates, and links. |
| `ranker.py` | Scores title and abstract term matches | Deterministic and explainable; easy to replace with reranking later. |
| `pdf_processing.py` | Downloads safely and extracts page text with PyMuPDF | Preserves page numbers for citations; enforces size, content-type, and PDF-header checks. |
| `chunker.py` | Breaks page text into overlapping section-aware chunks | Chunks fit embedding and model context while overlap avoids cutting a fact at a boundary. |
| `vector_store.py` | Creates normalized local embeddings and persists them in ChromaDB | Enables semantic retrieval without a paid API or a remote vector database. |
| `briefing.py` | Produces briefing and QA answers | Separates trusted metadata, deterministic limitation extraction, and model-authored analysis. |
| `session_store.py` | Saves paper, briefing, collection name, and chat history | Allows follow-up QA in a separate CLI command. |

## 3. Retrieval-augmented generation (RAG), in plain language

An embedding converts a piece of text into a list of numbers that captures semantic meaning. Similar meanings have vectors that are close together. The app embeds every chunk of the paper once. When the user asks a question, it embeds the question, finds the nearest chunks in ChromaDB, and supplies only those chunks to the model.

Cosine distance measures how dissimilar two normalized vectors are: smaller is more similar. This project uses a configurable maximum distance. If the best result is above that threshold, the program refuses instead of asking the model to guess. The threshold is a tradeoff: a strict threshold causes more safe refusals; a loose threshold may retrieve irrelevant evidence.

## 4. Grounding and anti-hallucination controls

There is no single perfect prompt that guarantees grounding, so the project uses several layers:

1. arXiv metadata overwrites any model-generated identity data.
2. PDF bytes are validated before parsing.
3. QA only receives retrieved chunks, never the whole internet or general chat history.
4. The distance gate refuses weak retrieval before generation.
5. The model must cite chunk IDs from an allow-list.
6. Application code removes invented citations and refuses an answer that has no valid citation.
7. Limitations are extracted from author-caveat sentences rather than inferred by the model.

An interviewer may ask why prompts alone are insufficient. The answer is that a prompt is a soft instruction, while metadata overwrite, retrieval thresholds, and citation allow-lists are deterministic controls enforced by code.

## 5. Failure paths

- Invalid input: classifier raises a readable validation error.
- Topic search returns no papers: graph enters `failed` with an actionable error.
- Download is oversized, not a PDF, or corrupt: it is rejected safely.
- Extracted text is too short: the graph creates an abstract-only briefing and deliberately does not enable detailed QA.
- Local Ollama service/model is unavailable: the user sees local setup guidance; there is no paid-cloud fallback.
- QA retrieval is weak: the agent returns the fixed insufficient-evidence response.

## 6. Four-minute video outline

**0:00–0:30 — goal.** “This is a local-first LangGraph agent that turns one arXiv paper into an executive briefing and supports evidence-grounded questions. It needs no paid API key.”

**0:30–1:20 — architecture.** Show the README graph. Explain the two input routes, then PDF parsing, chunks, embeddings, ChromaDB, briefing, and saved session.

**1:20–2:05 — run the digest.** Run `arxiv-digest digest 1706.03762`. Point out trusted arXiv identity fields and the author-stated limitation section.

**2:05–2:55 — supported QA.** Run the WMT evaluation question. Highlight the answer and the two citations, including page and section names.

**2:55–3:25 — refusal.** Ask for the authors' office address. Explain that the system refuses because no sufficiently relevant paper chunk exists.

**3:25–4:00 — tradeoffs.** Mention local privacy/cost advantages, initial model-download cost, deterministic keyword ranking as a deliberately simple choice, and OCR/reranking as production extensions.

## 7. Likely interview questions

**Why LangGraph instead of a linear script?** The requirement asks for explicit state, nodes, and edges. It also makes branching and failure routes visible and testable.

**Why use both an embedding threshold and model citations?** The threshold decides whether there is evidence worth answering from. Citation validation ensures the final answer points only to chunks actually retrieved. They solve different failure modes.

**Why overlap chunks?** A fact can begin at the end of one chunk and finish in the next. Overlap preserves that local context at the cost of a few duplicate embeddings.

**What happens with a scanned PDF?** PyMuPDF produces too little text, so the application labels it abstract-only and blocks detailed QA. OCR would be a future extension, not a hidden failure.

**Why not use an online API?** The assessment allows local models and asks to avoid paid keys. Ollama plus Qwen and sentence-transformers keep all inference local.

**What would you improve first?** Add a user choice for ranked topic candidates, a cross-encoder reranker, OCR for scanned papers, and evaluation cases that measure retrieval recall and citation faithfulness.
