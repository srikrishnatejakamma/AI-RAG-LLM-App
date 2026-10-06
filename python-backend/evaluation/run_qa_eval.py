from __future__ import annotations

import argparse
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("RAG_ADMIN_PASSWORD", "evaluation-only")

from app.ai import EmbeddingService, extractive_answer  # noqa: E402
from app.config import Settings  # noqa: E402
from app.domain import ChunkData, CollectionData, DocumentData  # noqa: E402
from app.retrieval import HybridRetriever  # noqa: E402
from app.text_pipeline import cosine  # noqa: E402


def reciprocal_rank(expected: set[str], ranked: list[str], limit: int) -> float:
    for rank, source_id in enumerate(ranked[:limit], start=1):
        if source_id in expected:
            return 1.0 / rank
    return 0.0


def recall_at_k(expected: set[str], ranked: list[str], limit: int) -> float:
    if not expected:
        return 1.0
    return len(expected.intersection(ranked[:limit])) / len(expected)


def ndcg_at_k(expected: set[str], ranked: list[str], limit: int) -> float:
    dcg = sum(1.0 / math.log2(rank + 1) for rank, source_id in enumerate(ranked[:limit], start=1) if source_id in expected)
    ideal_count = min(len(expected), limit)
    ideal_dcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_count + 1))
    return dcg / ideal_dcg if ideal_dcg else 0.0


def score_rankings(cases: list[dict[str, Any]], rankings: list[list[str]], limit: int) -> dict[str, float]:
    count = max(1, len(cases))
    return {
        f"recall@{limit}": sum(recall_at_k(set(case["relevant_sources"]), ranked, limit) for case, ranked in zip(cases, rankings)) / count,
        f"mrr@{limit}": sum(reciprocal_rank(set(case["relevant_sources"]), ranked, limit) for case, ranked in zip(cases, rankings)) / count,
        f"ndcg@{limit}": sum(ndcg_at_k(set(case["relevant_sources"]), ranked, limit) for case, ranked in zip(cases, rankings)) / count,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate local RAG source retrieval against labeled QA examples.")
    parser.add_argument("--provider", choices=("local", "transformers", "openai"), default="local")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--cases", type=Path, default=Path(__file__).with_name("qa_cases.json"))
    args = parser.parse_args()
    if args.top_k < 1:
        parser.error("--top-k must be at least 1")

    dataset = json.loads(args.cases.read_text(encoding="utf-8"))
    settings = Settings(embedding_provider=args.provider, admin_password="evaluation-only")
    embedding_service = EmbeddingService(settings)
    corpus = dataset["corpus"]
    vectors = embedding_service.embed_many([item["text"] for item in corpus])
    now = datetime.now(timezone.utc)
    collection = CollectionData(id="qa-eval", name="QA evaluation", created_at=now)
    for item, vector in zip(corpus, vectors):
        document_id = item["id"]
        collection.documents[document_id] = DocumentData(
            id=document_id,
            name=f"{document_id}.txt",
            checksum=document_id,
            uploaded_at=now,
            status="READY",
            error=None,
            chunks=[ChunkData(text=item["text"], index=1, page_number=None, vector=vector)],
        )

    entries = tuple(
        (document, chunk)
        for document in collection.documents.values()
        for chunk in document.chunks
    )
    retriever = HybridRetriever()
    dense_rankings: list[list[str]] = []
    hybrid_rankings: list[list[str]] = []
    answer_phrase_passes: list[bool] = []
    details: list[dict[str, Any]] = []

    for case in dataset["questions"]:
        question_vector = embedding_service.embed(case["question"])
        dense = []
        for document, chunk in entries:
            score = cosine(question_vector, chunk.vector)
            if score >= settings.minimum_score:
                dense.append((document.id, score))
        dense.sort(key=lambda row: row[1], reverse=True)
        dense = dense[: args.top_k]
        hybrid = retriever.retrieve(
            collection.id,
            entries,
            question_vector,
            case["question"],
            args.top_k,
            settings.minimum_score,
        )
        dense_ids = [source_id for source_id, _ in dense]
        hybrid_ids = [document.id for document, _, _ in hybrid]
        expected = set(case["relevant_sources"])
        answer = extractive_answer(case["question"], [
            f"[{document.name}] {chunk.text}" for document, chunk, _ in hybrid
        ])
        answer_pass = all(phrase.casefold() in answer.casefold() for phrase in case.get("answer_phrases", []))
        dense_rankings.append(dense_ids)
        hybrid_rankings.append(hybrid_ids)
        answer_phrase_passes.append(answer_pass)
        details.append({
            "question": case["question"],
            "expectedSources": sorted(expected),
            "expectedAnswerPhrases": case.get("answer_phrases", []),
            "denseTopK": dense_ids,
            "hybridTopK": hybrid_ids,
            "extractiveAnswer": answer,
            "answerPhrasePass": answer_pass,
        })

    report = {
        "provider": args.provider,
        "topK": args.top_k,
        "questions": len(dataset["questions"]),
        "denseOnly": score_rankings(dataset["questions"], dense_rankings, args.top_k),
        "hybrid": score_rankings(dataset["questions"], hybrid_rankings, args.top_k),
        "extractiveAnswerPhrasePassRate": sum(answer_phrase_passes) / max(1, len(answer_phrase_passes)),
        "cases": details,
    }
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
