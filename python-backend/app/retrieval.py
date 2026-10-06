from __future__ import annotations

import threading
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .domain import ChunkData, DocumentData
from .text_pipeline import cosine


@dataclass(frozen=True)
class _TfidfIndex:
    chunk_ids: tuple[int, ...]
    entries: tuple[tuple[DocumentData, ChunkData], ...]
    vectorizer: TfidfVectorizer | None
    matrix: Any | None
    char_vectorizer: TfidfVectorizer | None
    char_matrix: Any | None


class HybridRetriever:
    """Combines dense similarity with cached TF-IDF rankings using reciprocal rank fusion."""

    def __init__(self, max_cached_collections: int = 32, rrf_constant: int = 10, dense_weight: float = 1.0) -> None:
        self.max_cached_collections = max_cached_collections
        self.rrf_constant = rrf_constant
        self.dense_weight = dense_weight
        self._indexes: OrderedDict[str, _TfidfIndex] = OrderedDict()
        self._lock = threading.Lock()

    def retrieve(
        self,
        collection_id: str,
        entries: tuple[tuple[DocumentData, ChunkData], ...],
        query_vector: list[float],
        question: str,
        limit: int,
        minimum_dense_score: float,
    ) -> list[tuple[DocumentData, ChunkData, float]]:
        if not entries:
            return []

        index = self._get_index(collection_id, entries)
        dense_scores = [cosine(query_vector, chunk.vector) for _, chunk in entries]
        dense_order = sorted(range(len(entries)), key=lambda idx: (-dense_scores[idx], idx))
        fused: dict[int, float] = {}
        for rank, idx in enumerate(dense_order, start=1):
            if dense_scores[idx] >= minimum_dense_score:
                fused[idx] = fused.get(idx, 0.0) + self.dense_weight / (self.rrf_constant + rank)

        if index.vectorizer is not None and index.matrix is not None:
            query = index.vectorizer.transform([question])
            if query.nnz:
                lexical_scores = cosine_similarity(index.matrix, query).ravel()
                lexical_order = sorted(
                    (idx for idx, score in enumerate(lexical_scores) if score > 0),
                    key=lambda idx: (-lexical_scores[idx], idx),
                )
                for rank, idx in enumerate(lexical_order, start=1):
                    fused[idx] = fused.get(idx, 0.0) + 1.0 / (self.rrf_constant + rank)

        if index.char_vectorizer is not None and index.char_matrix is not None:
            query = index.char_vectorizer.transform([question])
            if query.nnz:
                char_scores = cosine_similarity(index.char_matrix, query).ravel()
                char_order = sorted(
                    (idx for idx, score in enumerate(char_scores) if score > 0),
                    key=lambda idx: (-char_scores[idx], idx),
                )
                for rank, idx in enumerate(char_order, start=1):
                    fused[idx] = fused.get(idx, 0.0) + 0.75 / (self.rrf_constant + rank)

        ranked = sorted(fused.items(), key=lambda row: (-row[1], row[0]))[:limit]
        return [(entries[idx][0], entries[idx][1], score) for idx, score in ranked]

    def _get_index(
        self,
        collection_id: str,
        entries: tuple[tuple[DocumentData, ChunkData], ...],
    ) -> _TfidfIndex:
        chunk_ids = tuple(id(chunk) for _, chunk in entries)
        with self._lock:
            cached = self._indexes.get(collection_id)
            if cached is not None and cached.chunk_ids == chunk_ids:
                self._indexes.move_to_end(collection_id)
                return cached

            vectorizer = TfidfVectorizer(
                lowercase=True,
                strip_accents="unicode",
                ngram_range=(1, 2),
                sublinear_tf=True,
            )
            char_vectorizer = TfidfVectorizer(
                analyzer="char_wb",
                lowercase=True,
                strip_accents="unicode",
                ngram_range=(3, 5),
                sublinear_tf=True,
            )
            try:
                texts = [chunk.text for _, chunk in entries]
                matrix = vectorizer.fit_transform(texts)
            except ValueError:
                vectorizer = None
                matrix = None
            try:
                char_matrix = char_vectorizer.fit_transform(texts)
            except ValueError:
                char_vectorizer = None
                char_matrix = None

            index = _TfidfIndex(chunk_ids, entries, vectorizer, matrix, char_vectorizer, char_matrix)
            self._indexes[collection_id] = index
            self._indexes.move_to_end(collection_id)
            while len(self._indexes) > self.max_cached_collections:
                self._indexes.popitem(last=False)
            return index
