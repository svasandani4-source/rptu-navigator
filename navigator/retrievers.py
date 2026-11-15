import re

import numpy as np
from rank_bm25 import BM25Okapi


def tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


class BM25Retriever:
    """Lexical retrieval. Strong for exact terms: 'Prüfungsamt', 'Rückmeldung', module codes."""
    name = "bm25"

    def __init__(self, chunks: list[dict]):
        self.chunks = chunks
        self.index = BM25Okapi([tokenize(c["text"]) for c in chunks])

    def search(self, query: str, k: int = 5) -> list[dict]:
        scores = self.index.get_scores(tokenize(query))
        return [self.chunks[i] for i in np.argsort(-scores)[:k]]


class DenseRetriever:
    """Semantic, multilingual retrieval. Handles 'how do I enroll?' vs 'Immatrikulation' style gaps."""
    name = "dense"

    def __init__(self, chunks: list[dict], model_name: str):
        from sentence_transformers import SentenceTransformer

        self.chunks = chunks
        self.model = SentenceTransformer(model_name)
        self.emb = self.model.encode([c["text"] for c in chunks], normalize_embeddings=True, show_progress_bar=False)

    def search(self, query: str, k: int = 5) -> list[dict]:
        q = self.model.encode([query], normalize_embeddings=True)[0]
        scores = (self.emb * q).sum(axis=1)
        return [self.chunks[i] for i in np.argsort(-scores)[:k]]


def rrf(rankings: list[list[dict]], top: int = 5, c: int = 60) -> list[dict]:
    """Reciprocal Rank Fusion: score = sum over rankers of 1 / (c + rank)."""
    scores: dict[str, float] = {}
    by_id: dict[str, dict] = {}
    for ranking in rankings:
        for rank, chunk in enumerate(ranking):
            by_id[chunk["id"]] = chunk
            scores[chunk["id"]] = scores.get(chunk["id"], 0.0) + 1.0 / (c + rank + 1)
    return [by_id[i] for i in sorted(scores, key=scores.get, reverse=True)[:top]]


class HybridRetriever:
    name = "hybrid_rrf"

    def __init__(self, bm25: BM25Retriever, dense: DenseRetriever, pool: int = 20):
        self.bm25, self.dense, self.pool = bm25, dense, pool

    def search(self, query: str, k: int = 5) -> list[dict]:
        return rrf([self.bm25.search(query, self.pool), self.dense.search(query, self.pool)], top=k)
