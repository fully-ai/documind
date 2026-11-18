"""Tests for DocuMind document extraction, indexing, hybrid search, and RAG."""

import json
from pathlib import Path

import httpx
from fastapi.testclient import TestClient

from documind.app import create_app
from documind.config import Settings


def _make_minimal_pdf(text: str) -> bytes:
    """Construct a valid single-page PDF containing extractable ASCII text."""
    stream = f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET".encode("latin1")
    objs = [
        b"1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj\n",
        b"2 0 obj << /Type /Pages /Kids [3 0 R] /Count 1 >> endobj\n",
        b"3 0 obj << /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        b"/Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >> endobj\n",
        b"4 0 obj << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> endobj\n",
        f"5 0 obj << /Length {len(stream)} >> stream\n".encode("latin1") + stream + b"\nendstream endobj\n",
    ]
    out = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for obj in objs:
        offsets.append(len(out))
        out.extend(obj)
    xref_pos = len(out)
    out.extend(f"xref\n0 {len(offsets)}\n0000000000 65535 f \n".encode("latin1"))
    for off in offsets[1:]:
        out.extend(f"{off:010d} 00000 n \n".encode("latin1"))
    out.extend(
        f"trailer << /Size {len(offsets)} /Root 1 0 R >>\nstartxref\n{xref_pos}\n%%EOF\n".encode("latin1")
    )
    return bytes(out)


def _mock_ollama_transport(available: bool = True) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        if not available:
            raise httpx.ConnectError("offline")
        if request.url.path == "/api/embeddings":
            prompt = json.loads(request.content)["prompt"].lower()
            vec = [1.0, 0.0] if "kubernetes" in prompt or "cluster" in prompt else [0.0, 1.0]
            return httpx.Response(200, json={"embedding": vec})
        if request.url.path == "/api/chat":
            return httpx.Response(
                200,
                json={"message": {"content": "Control plane manages etcd [Source 1]."}},
            )
        return httpx.Response(404)

    return httpx.MockTransport(handler)


def test_pdf_and_markdown_indexing_tfidf_fallback(tmp_path: Path) -> None:
    cfg = Settings(data_dir=tmp_path / "dm")
    app = create_app(cfg, ollama_transport=_mock_ollama_transport(available=False))

    with TestClient(app) as tc:
        col = tc.post("/api/collections", json={"name": "ops", "description": "Runbooks"})
        assert col.status_code == 201
        col_id = col.json()["id"]

        pdf_bytes = _make_minimal_pdf("Kubernetes etcd backup procedure runs daily")
        up_pdf = tc.post(
            "/api/documents",
            data={"collection_id": col_id},
            files={"file": ("runbook.pdf", pdf_bytes, "application/pdf")},
        )
        assert up_pdf.status_code == 201
        assert up_pdf.json()["page_count"] == 1

        up_md = tc.post(
            "/api/documents",
            data={"collection_id": "default"},
            files={"file": ("notes.md", b"# Gardening\nWater tomatoes twice a week.", "text/markdown")},
        )
        assert up_md.status_code == 201

        search = tc.post("/api/search", json={"query": "etcd backup", "collection_id": col_id, "top_k": 3})
        assert search.status_code == 200
        body = search.json()
        assert body["mode"] == "tfidf"
        assert len(body["results"]) == 1
        assert body["results"][0]["filename"] == "runbook.pdf"

        rag = tc.post("/api/rag", json={"query": "etcd backup", "collection_id": col_id, "top_k": 3})
        assert rag.status_code == 200
        assert rag.json()["used_ollama"] is False
        assert "Kubernetes" in rag.json()["answer"]

        hist = tc.get("/api/history").json()["history"]
        assert len(hist) == 2


def test_ollama_embeddings_and_rag(tmp_path: Path) -> None:
    cfg = Settings(data_dir=tmp_path / "dm2")
    app = create_app(cfg, ollama_transport=_mock_ollama_transport(available=True))

    with TestClient(app) as tc:
        up = tc.post(
            "/api/documents",
            data={"collection_id": "default"},
            files={"file": ("k8s.txt", b"Kubernetes cluster control plane runs etcd.", "text/plain")},
        )
        assert up.status_code == 201

        rag = tc.post("/api/rag", json={"query": "How does cluster work?", "top_k": 2})
        assert rag.status_code == 200
        data = rag.json()
        assert data["used_ollama"] is True
        assert data["retrieval_mode"] == "embedding"
        assert "[Source 1]" in data["answer"]

        doc_id = up.json()["id"]
        assert tc.delete(f"/api/documents/{doc_id}").status_code == 200
        assert tc.delete(f"/api/documents/{doc_id}").status_code == 404

        bad_up = tc.post(
            "/api/documents",
            data={"collection_id": "default"},
            files={"file": ("malware.bin", b"\x00\x01\x02", "application/octet-stream")},
        )
        assert bad_up.status_code == 400


def test_deduplication_chunk_inspector_reindex_rrf_and_batch_ingest(tmp_path: Path) -> None:
    import asyncio

    from documind.ingest import ingest_directory

    cfg = Settings(data_dir=tmp_path / "dm3")
    app = create_app(cfg, ollama_transport=_mock_ollama_transport(available=False))

    with TestClient(app) as tc:
        payload = b"Kubernetes etcd snapshot and restore runbook."
        r1 = tc.post(
            "/api/documents",
            data={"collection_id": "default"},
            files={"file": ("k8s.md", payload, "text/markdown")},
        )
        assert r1.status_code == 201
        assert r1.json()["deduplicated"] is False

        # Re-uploading identical content in same collection deduplicates cleanly
        r2 = tc.post(
            "/api/documents",
            data={"collection_id": "default"},
            files={"file": ("k8s_copy.md", payload, "text/markdown")},
        )
        assert r2.status_code == 200
        assert r2.json()["deduplicated"] is True
        assert r2.json()["id"] == r1.json()["id"]

        doc_id = r1.json()["id"]
        chunks_res = tc.get(f"/api/documents/{doc_id}/chunks")
        assert chunks_res.status_code == 200
        assert len(chunks_res.json()["chunks"]) >= 1
        assert chunks_res.json()["chunks"][0]["has_embedding"] == 0

    # Re-instantiate with Ollama available, reindex existing chunks, and test hybrid RRF search
    app_online = create_app(cfg, ollama_transport=_mock_ollama_transport(available=True))
    with TestClient(app_online) as tc2:
        reidx = tc2.post(f"/api/documents/{doc_id}/reindex")
        assert reidx.status_code == 200
        assert reidx.json()["embeddings_updated"] >= 1

        hyb = tc2.post("/api/search", json={"query": "etcd snapshot", "mode": "hybrid", "top_k": 3})
        assert hyb.status_code == 200
        assert hyb.json()["mode"] == "hybrid"
        assert len(hyb.json()["results"]) == 1

    # Batch directory ingestion CLI helper
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "guide.txt").write_text("Systemd journal rotation guide.", encoding="utf-8")
    (corpus / "notes.md").write_text("# Postgres\nVacuum and analyze.", encoding="utf-8")
    (corpus / "ignored.bin").write_bytes(b"\x00\x01\x02")
    cfg_batch = Settings(data_dir=tmp_path / "dm_batch")
    res1 = asyncio.run(ingest_directory(corpus, collection_name="ops", settings=cfg_batch))
    assert res1 == {"indexed": 2, "deduplicated": 0, "skipped": 0}
    res2 = asyncio.run(ingest_directory(corpus, collection_name="ops", settings=cfg_batch))
    assert res2 == {"indexed": 0, "deduplicated": 2, "skipped": 0}
