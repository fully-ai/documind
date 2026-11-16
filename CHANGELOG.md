# Changelog

## [0.1.0] - 2026-11-16

### Added
- PDF and UTF-8 text/Markdown document extraction with page-level metadata and overlapping chunking.
- Hybrid retrieval: Ollama dense embeddings + cosine similarity, with built-in TF-IDF cosine fallback.
- Grounded RAG endpoint (`/api/rag`) with source citations, document collections, and search history.
