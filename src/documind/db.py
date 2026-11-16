"""Async SQLite repository for collections, documents, chunks, embeddings, and search history."""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import aiosqlite


def utcnow_iso() -> str:
    return datetime.now(UTC).isoformat()


SCHEMA_SQL = """
PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS collections (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    description TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS documents (
    id TEXT PRIMARY KEY,
    collection_id TEXT NOT NULL,
    filename TEXT NOT NULL,
    title TEXT NOT NULL,
    author TEXT NOT NULL DEFAULT '',
    mime_type TEXT NOT NULL,
    page_count INTEGER NOT NULL,
    size_bytes INTEGER NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY (collection_id) REFERENCES collections(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS chunks (
    id TEXT PRIMARY KEY,
    document_id TEXT NOT NULL,
    collection_id TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    page_number INTEGER NOT NULL,
    content TEXT NOT NULL,
    embedding_json TEXT,
    FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_chunks_collection ON chunks(collection_id);

CREATE TABLE IF NOT EXISTS search_history (
    id TEXT PRIMARY KEY,
    collection_id TEXT,
    query TEXT NOT NULL,
    mode TEXT NOT NULL,
    result_count INTEGER NOT NULL,
    created_at TEXT NOT NULL
);
"""


class DocuDatabase:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self._conn: aiosqlite.Connection | None = None

    async def connect(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = await aiosqlite.connect(self.db_path)
        self._conn.row_factory = aiosqlite.Row
        await self._conn.executescript(SCHEMA_SQL)
        await self._conn.execute(
            "INSERT OR IGNORE INTO collections (id, name, description, created_at) VALUES (?, ?, ?, ?)",
            ("default", "default", "Default document collection", utcnow_iso()),
        )
        await self._conn.commit()

    async def close(self) -> None:
        if self._conn is not None:
            await self._conn.close()
            self._conn = None

    @property
    def conn(self) -> aiosqlite.Connection:
        if self._conn is None:
            raise RuntimeError("Database not initialized")
        return self._conn

    async def list_collections(self) -> list[dict[str, Any]]:
        cur = await self.conn.execute("SELECT * FROM collections ORDER BY name ASC")
        return [dict(r) for r in await cur.fetchall()]

    async def create_collection(self, name: str, description: str) -> dict[str, Any]:
        cid = uuid.uuid4().hex
        now = utcnow_iso()
        await self.conn.execute(
            "INSERT INTO collections (id, name, description, created_at) VALUES (?, ?, ?, ?)",
            (cid, name, description, now),
        )
        await self.conn.commit()
        return {"id": cid, "name": name, "description": description, "created_at": now}

    async def get_collection(self, collection_id: str) -> dict[str, Any] | None:
        cur = await self.conn.execute("SELECT * FROM collections WHERE id = ?", (collection_id,))
        row = await cur.fetchone()
        return dict(row) if row else None

    async def add_document(
        self,
        collection_id: str,
        filename: str,
        title: str,
        author: str,
        mime_type: str,
        page_count: int,
        size_bytes: int,
        chunks: list[dict[str, Any]],
    ) -> dict[str, Any]:
        doc_id = uuid.uuid4().hex
        now = utcnow_iso()
        await self.conn.execute(
            """
            INSERT INTO documents (
                id, collection_id, filename, title, author, mime_type, page_count, size_bytes, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (doc_id, collection_id, filename, title, author, mime_type, page_count, size_bytes, now),
        )
        for ch in chunks:
            emb_json = json.dumps(ch["embedding"]) if ch.get("embedding") is not None else None
            await self.conn.execute(
                """
                INSERT INTO chunks (id, document_id, collection_id, chunk_index, page_number, content, embedding_json)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    uuid.uuid4().hex,
                    doc_id,
                    collection_id,
                    ch["chunk_index"],
                    ch["page_number"],
                    ch["content"],
                    emb_json,
                ),
            )
        await self.conn.commit()
        return {
            "id": doc_id,
            "collection_id": collection_id,
            "filename": filename,
            "title": title,
            "author": author,
            "mime_type": mime_type,
            "page_count": page_count,
            "size_bytes": size_bytes,
            "chunk_count": len(chunks),
            "created_at": now,
        }

    async def list_documents(self, collection_id: str | None = None) -> list[dict[str, Any]]:
        if collection_id:
            cur = await self.conn.execute(
                "SELECT * FROM documents WHERE collection_id = ? ORDER BY created_at DESC",
                (collection_id,),
            )
        else:
            cur = await self.conn.execute("SELECT * FROM documents ORDER BY created_at DESC")
        return [dict(r) for r in await cur.fetchall()]

    async def delete_document(self, doc_id: str) -> bool:
        cur = await self.conn.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
        await self.conn.commit()
        return cur.rowcount > 0

    async def load_chunks(self, collection_id: str | None = None) -> list[dict[str, Any]]:
        sql = """
            SELECT c.id, c.document_id, c.collection_id, c.chunk_index, c.page_number,
                   c.content, c.embedding_json, d.filename, d.title
            FROM chunks c
            JOIN documents d ON d.id = c.document_id
        """
        params: tuple[Any, ...] = ()
        if collection_id:
            sql += " WHERE c.collection_id = ?"
            params = (collection_id,)
        cur = await self.conn.execute(sql, params)
        rows = await cur.fetchall()
        out = []
        for r in rows:
            item = dict(r)
            item["embedding"] = json.loads(item.pop("embedding_json")) if item["embedding_json"] else None
            out.append(item)
        return out

    async def record_search(self, collection_id: str | None, query: str, mode: str, result_count: int) -> None:
        await self.conn.execute(
            """
            INSERT INTO search_history (id, collection_id, query, mode, result_count, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (uuid.uuid4().hex, collection_id, query, mode, result_count, utcnow_iso()),
        )
        await self.conn.commit()

    async def list_search_history(self, limit: int = 50) -> list[dict[str, Any]]:
        cur = await self.conn.execute(
            "SELECT * FROM search_history ORDER BY created_at DESC LIMIT ?",
            (limit,),
        )
        return [dict(r) for r in await cur.fetchall()]
