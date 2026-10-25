"""Dense, BM25, and hybrid retrieval with optional cross-encoder re-ranking."""
from __future__ import annotations
import os
import numpy as np
from .config import RetrievalCfg
from .schemas import Chunk


class Retriever:
    def __init__(self, chunks: list[Chunk], cfg: RetrievalCfg):
        self.chunks = chunks
        self.cfg = cfg
        self._texts = [c.text for c in chunks]
        self._dense = self._bm25 = self._reranker = None
        if cfg.mode == "none":
            return  # control condition: build no index, retrieve nothing
        if cfg.mode in ("dense", "hybrid"):
            from sentence_transformers import SentenceTransformer
            import faiss
            self._embedder = SentenceTransformer(os.getenv("EMBEDDING_MODEL", "BAAI/bge-base-en-v1.5"))
            emb = self._embedder.encode(self._texts, normalize_embeddings=True, show_progress_bar=True)
            self._dense = faiss.IndexFlatIP(emb.shape[1])
            self._dense.add(np.asarray(emb, dtype="float32"))
        if cfg.mode in ("bm25", "hybrid"):
            from rank_bm25 import BM25Okapi
            self._bm25 = BM25Okapi([t.lower().split() for t in self._texts])
        if cfg.rerank:
            from sentence_transformers import CrossEncoder
            self._reranker = CrossEncoder(cfg.rerank_model or "cross-encoder/ms-marco-MiniLM-L-6-v2")

    def _dense_scores(self, query: str) -> np.ndarray:
        q = self._embedder.encode([query], normalize_embeddings=True)
        scores = np.zeros(len(self.chunks), dtype="float32")
        D, I = self._dense.search(np.asarray(q, dtype="float32"), len(self.chunks))
        scores[I[0]] = D[0]
        return scores

    def _bm25_scores(self, query: str) -> np.ndarray:
        s = np.asarray(self._bm25.get_scores(query.lower().split()), dtype="float32")
        return s / (s.max() + 1e-9)

    def retrieve(self, query: str) -> list[Chunk]:
        if self.cfg.mode == "none":
            return []
        if self.cfg.mode == "dense":
            scores = self._dense_scores(query)
        elif self.cfg.mode == "bm25":
            scores = self._bm25_scores(query)
        else:
            a = self.cfg.hybrid_alpha
            scores = a * self._dense_scores(query) + (1 - a) * self._bm25_scores(query)

        pool = self.cfg.top_k * 4 if self._reranker else self.cfg.top_k
        idx = np.argsort(-scores)[:pool]
        cands = [self.chunks[i] for i in idx]
        if self._reranker:
            rs = self._reranker.predict([(query, c.text) for c in cands])
            cands = [c for _, c in sorted(zip(rs, cands), key=lambda x: -x[0])]
        return cands[: self.cfg.top_k]
