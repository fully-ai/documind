"""FastAPI backend for DocuMind document indexing, search, and RAG."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Any

import httpx
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from documind.config import Settings
from documind.db import DocuDatabase
from documind.extractor import ExtractionError, chunk_pages, extract_document
from documind.rag import OllamaRAGClient, cosine_similarity, score_chunks_tfidf

STATIC_DIR = Path(__file__).parent / "static"


class CollectionCreate(BaseModel):
    name: str = Field(min_length=1, max_length=80)
    description: str = Field(default="", max_length=300)


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=1000)
    collection_id: str | None = None
    top_k: int = Field(default=5, ge=1, le=20)


def create_app(
    settings: Settings | None = None,
    ollama_transport: httpx.AsyncBaseTransport | None = None,
) -> FastAPI:
    cfg = settings or Settings()
    cfg.ensure_directories()
    db = DocuDatabase(cfg.db_path)
    rag = OllamaRAGClient(
        base_url=cfg.ollama_base_url,
        embed_model=cfg.embed_model,
        chat_model=cfg.chat_model,
        transport=ollama_transport,
    )

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        await db.connect()
        try:
            yield
        finally:
            await rag.close()
            await db.close()

    app = FastAPI(title="DocuMind", version="0.1.0", lifespan=lifespan)

    @app.get("/", include_in_schema=False)
    async def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/api/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/api/collections")
    async def list_collections() -> dict[str, Any]:
        return {"collections": await db.list_collections()}

    @app.post("/api/collections", status_code=201)
    async def create_collection(req: CollectionCreate) -> dict[str, Any]:
        try:
            return await db.create_collection(req.name, req.description)
        except Exception as exc:
            raise HTTPException(status_code=409, detail="Collection already exists") from exc

    @app.post("/api/documents", status_code=201)
    async def upload_document(
        file: Annotated[UploadFile, File()],
        collection_id: Annotated[str, Form()] = "default",
    ) -> dict[str, Any]:
        if await db.get_collection(collection_id) is None:
            raise HTTPException(status_code=404, detail="Collection not found")
        content = await file.read(cfg.max_file_bytes + 1)
        if not content:
            raise HTTPException(status_code=400, detail="Empty file")
        if len(content) > cfg.max_file_bytes:
            raise HTTPException(status_code=413, detail="File exceeds max size limit")
        try:
            doc = extract_document(file.filename or "document.txt", content)
        except ExtractionError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        chunks = chunk_pages(doc.pages, chunk_size=cfg.chunk_size, overlap=cfg.chunk_overlap)
        for ch in chunks:
            ch["embedding"] = await rag.embed(ch["content"])

        return await db.add_document(
            collection_id=collection_id,
            filename=doc.filename,
            title=doc.title,
            author=doc.author,
            mime_type=doc.mime_type,
            page_count=doc.page_count,
            size_bytes=len(content),
            chunks=chunks,
        )

    @app.get("/api/documents")
    async def list_documents(collection_id: str | None = None) -> dict[str, Any]:
        return {"documents": await db.list_documents(collection_id)}

    @app.delete("/api/documents/{doc_id}")
    async def delete_document(doc_id: str) -> dict[str, str]:
        if not await db.delete_document(doc_id):
            raise HTTPException(status_code=404, detail="Document not found")
        return {"status": "deleted"}

    async def _rank_chunks(query: str, collection_id: str | None, top_k: int) -> tuple[str, list[dict[str, Any]]]:
        chunks = await db.load_chunks(collection_id)
        if not chunks:
            return "tfidf", []
        q_emb = await rag.embed(query)
        with_emb = [c for c in chunks if c["embedding"] is not None]
        if q_emb is not None and len(with_emb) == len(chunks):
            scored = [
                (cosine_similarity(q_emb, c["embedding"]), c)
                for c in chunks
            ]
            scored = [s for s in scored if s[0] > 0.0]
            scored.sort(key=lambda x: x[0], reverse=True)
            mode = "embedding"
        else:
            scored = score_chunks_tfidf(query, chunks)
            mode = "tfidf"

        results = []
        for score, ch in scored[:top_k]:
            results.append(
                {
                    "score": round(score, 4),
                    "document_id": ch["document_id"],
                    "filename": ch["filename"],
                    "title": ch["title"],
                    "page_number": ch["page_number"],
                    "chunk_index": ch["chunk_index"],
                    "content": ch["content"],
                }
            )
        return mode, results

    @app.post("/api/search")
    async def search(req: SearchRequest) -> dict[str, Any]:
        mode, results = await _rank_chunks(req.query, req.collection_id, req.top_k)
        await db.record_search(req.collection_id, req.query, mode, len(results))
        return {"mode": mode, "results": results}

    @app.post("/api/rag")
    async def rag_answer(req: SearchRequest) -> dict[str, Any]:
        mode, results = await _rank_chunks(req.query, req.collection_id, req.top_k)
        await db.record_search(req.collection_id, req.query, f"rag:{mode}", len(results))
        answer, used_llm = await rag.generate_answer(req.query, results)
        return {
            "answer": answer,
            "used_ollama": used_llm,
            "retrieval_mode": mode,
            "sources": results,
        }

    @app.get("/api/history")
    async def history() -> dict[str, Any]:
        return {"history": await db.list_search_history()}

    if STATIC_DIR.is_dir():
        app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    return app
