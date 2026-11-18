"""Local Ollama embedding/RAG client plus deterministic TF-IDF cosine scorer."""

from __future__ import annotations

import math
from collections import Counter
from typing import Any

import httpx

from documind.extractor import tokenize


def cosine_similarity(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def score_chunks_tfidf(query: str, chunks: list[dict[str, Any]]) -> list[tuple[float, dict[str, Any]]]:
    """Compute TF-IDF cosine similarity across candidate chunks (zero external deps)."""
    q_tokens = tokenize(query)
    if not q_tokens or not chunks:
        return []

    doc_tokens = [tokenize(c["content"]) for c in chunks]
    num_docs = len(doc_tokens)
    df: Counter[str] = Counter()
    for toks in doc_tokens:
        for tok in set(toks):
            df[tok] += 1

    def tfidf_vec(toks: list[str]) -> dict[str, float]:
        tf = Counter(toks)
        total = max(1, len(toks))
        return {
            t: (cnt / total) * (math.log((1 + num_docs) / (1 + df.get(t, 0))) + 1.0)
            for t, cnt in tf.items()
        }

    q_vec = tfidf_vec(q_tokens)
    q_norm = math.sqrt(sum(v * v for v in q_vec.values())) or 1.0

    scored: list[tuple[float, dict[str, Any]]] = []
    for chunk, toks in zip(chunks, doc_tokens, strict=True):
        d_vec = tfidf_vec(toks)
        d_norm = math.sqrt(sum(v * v for v in d_vec.values())) or 1.0
        dot = sum(q_vec[t] * d_vec.get(t, 0.0) for t in q_vec)
        sim = dot / (q_norm * d_norm)
        if sim > 0.0:
            scored.append((sim, chunk))

    scored.sort(key=lambda item: item[0], reverse=True)
    return scored


def rank_hybrid_rrf(
    query: str,
    chunks: list[dict[str, Any]],
    query_embedding: list[float] | None,
    k: int = 60,
) -> list[tuple[float, dict[str, Any]]]:
    """Combine dense embedding cosine rank and TF-IDF lexical rank via Reciprocal Rank Fusion."""
    if not chunks:
        return []
    tfidf_ranked = score_chunks_tfidf(query, chunks)
    emb_ranked: list[tuple[float, dict[str, Any]]] = []
    if query_embedding is not None:
        for ch in chunks:
            emb = ch.get("embedding")
            if emb is not None:
                sim = cosine_similarity(query_embedding, emb)
                if sim > 0.0:
                    emb_ranked.append((sim, ch))
        emb_ranked.sort(key=lambda item: item[0], reverse=True)

    if not emb_ranked:
        return tfidf_ranked

    scores: dict[str, float] = {}
    by_id: dict[str, dict[str, Any]] = {}
    for rank_idx, (_, ch) in enumerate(tfidf_ranked, start=1):
        cid = str(ch["id"])
        by_id[cid] = ch
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank_idx)
    for rank_idx, (_, ch) in enumerate(emb_ranked, start=1):
        cid = str(ch["id"])
        by_id[cid] = ch
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank_idx)

    fused = [(score, by_id[cid]) for cid, score in scores.items()]
    fused.sort(key=lambda item: item[0], reverse=True)
    return fused


class OllamaRAGClient:
    def __init__(
        self,
        base_url: str,
        embed_model: str,
        chat_model: str,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.embed_model = embed_model
        self.chat_model = chat_model
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=httpx.Timeout(30.0, connect=2.0),
            transport=transport,
        )

    async def close(self) -> None:
        await self._client.aclose()

    async def embed(self, text: str) -> list[float] | None:
        try:
            resp = await self._client.post(
                "/api/embeddings",
                json={"model": self.embed_model, "prompt": text},
            )
            if resp.status_code != 200:
                return None
            vec = resp.json().get("embedding")
            if isinstance(vec, list) and vec:
                return [float(x) for x in vec]
            return None
        except Exception:
            return None

    async def generate_answer(self, question: str, contexts: list[dict[str, Any]]) -> tuple[str, bool]:
        """Generate grounded answer with source citations via Ollama, or return extractive fallback."""
        context_block = "\n\n".join(
            f"[Source {i+1}: {c['filename']} p.{c['page_number']}]\n{c['content']}"
            for i, c in enumerate(contexts)
        )
        prompt = (
            "Answer the user question using ONLY the provided sources. "
            "Cite sources using [Source N]. If the sources do not contain the answer, say so.\n\n"
            f"{context_block}\n\nQuestion: {question}"
        )
        try:
            resp = await self._client.post(
                "/api/chat",
                json={
                    "model": self.chat_model,
                    "messages": [{"role": "user", "content": prompt}],
                    "stream": False,
                },
            )
            if resp.status_code == 200:
                content = resp.json().get("message", {}).get("content", "")
                if content:
                    return str(content), True
        except Exception:
            pass

        if not contexts:
            return "No matching passages were found in the indexed documents.", False
        summary = " ".join(c["content"] for c in contexts[:2])
        return f"Top matching excerpts (extractive fallback; Ollama unavailable): {summary}", False
