# Changelog

## [0.3.0] - 2026-11-18

### Added
- **SHA-256 Deduplication & Schema Migration**: Prevents duplicate document uploads inside the same collection (`deduplicated: true`) and migrates legacy databases cleanly.
- **Chunk Inspector & Re-indexing**: Added `GET /api/documents/{id}/chunks`, `POST /api/documents/{id}/reindex`, and UI chunk inspector modal.
- **Hybrid RRF Search & Batch CLI**: Added Reciprocal Rank Fusion (`mode="hybrid"`) combining dense embedding and TF-IDF ranks, plus `documind-ingest` recursive directory CLI and `scripts/{install,dev,test}.{sh,ps1}`.

## [0.2.0] - 2026-11-16

### Added
- Collection creation, document deletion buttons, and search history panel in the web UI.
- Tests for document deletion and unsupported binary upload rejection.
- Comprehensive README with Mermaid architecture diagram, API reference, and known limitations.

## [0.1.0] - 2026-11-16

### Added
- PDF and UTF-8 text/Markdown document extraction with page-level metadata and overlapping chunking.
- Hybrid retrieval: Ollama dense embeddings + cosine similarity, with built-in TF-IDF cosine fallback.
- Grounded RAG endpoint (`/api/rag`) with source citations, document collections, and search history.
