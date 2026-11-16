# DocuMind

Local document indexing, semantic/TF-IDF hybrid search, and Ollama-backed RAG platform with source citations.

## Features

- **PDF & Text Indexing**: Extracts text and metadata from `.pdf`, `.txt`, `.md`, `.py`, `.json`, `.yaml`, and `.log` files with overlap-aware chunking.
- **Hybrid Retrieval**: Uses Ollama `/api/embeddings` when available and automatically falls back to built-in TF-IDF cosine similarity when running offline.
- **RAG with Citations**: Builds grounded answers referencing `[Source N]` filename and page numbers.
- **Collections & Search History**: Organize documents into isolated collections and audit past queries.
- **Web UI & Docker Support**: Includes a browser interface and `compose.yaml`.

## Installation & Running

```bash
git clone https://github.com/fully-ai/documind.git
cd documind
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
documind
```

## Testing

```bash
ruff check src tests
mypy src
pytest -v
```

## License

Apache License 2.0 — see [LICENSE](LICENSE).
