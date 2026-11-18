"""Batch directory ingestion CLI for DocuMind."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import sys
from pathlib import Path

from documind.config import Settings
from documind.db import DocuDatabase
from documind.extractor import ExtractionError, chunk_pages, extract_document
from documind.rag import OllamaRAGClient

DEFAULT_EXTS = {".txt", ".md", ".markdown", ".pdf"}


async def ingest_directory(
    directory: Path,
    collection_name: str = "default",
    settings: Settings | None = None,
    extensions: set[str] | None = None,
) -> dict[str, int]:
    cfg = settings or Settings()
    cfg.ensure_directories()
    exts = extensions or DEFAULT_EXTS
    db = DocuDatabase(cfg.db_path)
    rag = OllamaRAGClient(cfg.ollama_base_url, cfg.embed_model, cfg.chat_model)

    summary = {"indexed": 0, "deduplicated": 0, "skipped": 0}
    await db.connect()
    try:
        cols = await db.list_collections()
        match = next((c for c in cols if c["name"] == collection_name or c["id"] == collection_name), None)
        if match is None:
            match = await db.create_collection(collection_name, f"Batch ingested from {directory.name}")
        collection_id = str(match["id"])

        for path in sorted(directory.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in exts:
                continue
            raw = path.read_bytes()
            if not raw or len(raw) > cfg.max_file_bytes:
                summary["skipped"] += 1
                continue
            sha = hashlib.sha256(raw).hexdigest()
            if await db.find_document_by_hash(collection_id, sha) is not None:
                summary["deduplicated"] += 1
                continue
            try:
                doc = extract_document(path.name, raw)
            except ExtractionError:
                summary["skipped"] += 1
                continue

            chunks = chunk_pages(doc.pages, chunk_size=cfg.chunk_size, overlap=cfg.chunk_overlap)
            for ch in chunks:
                ch["embedding"] = await rag.embed(ch["content"])

            await db.add_document(
                collection_id=collection_id,
                filename=doc.filename,
                title=doc.title,
                author=doc.author,
                mime_type=doc.mime_type,
                page_count=doc.page_count,
                size_bytes=len(raw),
                chunks=chunks,
                content_sha256=sha,
            )
            summary["indexed"] += 1
    finally:
        await rag.close()
        await db.close()
    return summary


def cli(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="documind-ingest", description="Batch ingest a local directory into DocuMind")
    parser.add_argument("directory", type=Path, help="Directory to scan recursively")
    parser.add_argument("--collection", default="default", help="Target collection name or ID")
    parser.add_argument("--json", action="store_true", help="Print JSON summary")
    args = parser.parse_args(argv)

    if not args.directory.is_dir():
        print(f"Directory not found: {args.directory}", file=sys.stderr)
        return 1

    res = asyncio.run(ingest_directory(args.directory, collection_name=args.collection))
    if args.json:
        print(json.dumps(res, indent=2))
    else:
        print(f"Indexed={res['indexed']} Deduplicated={res['deduplicated']} Skipped={res['skipped']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(cli())
