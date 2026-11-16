# Architecture — DocuMind

## Components

- `src/documind/extractor.py`: Extracts text and metadata from PDF (`pypdf`) and UTF-8 text/Markdown files and splits content into overlapping chunks preserving page numbers.
- `src/documind/rag.py`: Queries Ollama `/api/embeddings` for dense cosine similarity retrieval, automatically falling back to built-in TF-IDF cosine scoring when Ollama is unavailable, and builds `[Source N]` cited prompts for `/api/chat`.
- `src/documind/db.py`: SQLite persistence for collections, document metadata, chunks/embeddings, and search history.
- `src/documind/app.py`: FastAPI REST endpoints and web UI.
