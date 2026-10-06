from __future__ import annotations

import hashlib
import math
import re


TOKEN_RE = re.compile(r"[\w\d]{2,}", re.UNICODE)


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    # Keep source line boundaries so section titles remain distinguishable after indexing.
    normalized = re.sub(r"[ \t\r\f\v]+", " ", text)
    normalized = re.sub(r" *\n *", "\n", normalized).strip()
    if not normalized:
        return []
    if chunk_size < 1:
        raise ValueError("Chunk size must be positive")
    words = normalized.split(" ")
    result: list[str] = []
    start = 0
    while start < len(words):
        end = start
        char_count = 0
        while end < len(words):
            additional = len(words[end]) + (1 if end > start else 0)
            if end > start and char_count + additional > chunk_size:
                break
            if end == start and len(words[end]) > chunk_size:
                break
            char_count += additional
            end += 1
        if end == start:
            word = words[start]
            result.append(word[:chunk_size])
            words[start] = word[chunk_size:]
            if not words[start]:
                start += 1
            continue
        result.append(" ".join(words[start:end]))
        if end >= len(words):
            break
        next_start = end
        if overlap > 0:
            overlap_chars = 0
            overlap_words = 0
            for word in reversed(words[start:end]):
                additional = len(word) + (1 if overlap_words else 0)
                if overlap_words and overlap_chars + additional > overlap:
                    break
                overlap_chars += additional
                overlap_words += 1
            next_start = max(start + 1, end - overlap_words)
        start = next_start
    return result


def checksum(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def embed_local(text: str, dimensions: int = 1536) -> list[float]:
    vector = [0.0] * dimensions
    for token in TOKEN_RE.findall(text.lower()):
        token_hash = int.from_bytes(
            hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest(),
            byteorder="big",
        )
        index = token_hash % dimensions
        vector[index] += -1.0 if (token_hash & (1 << 63)) else 1.0
    norm = math.sqrt(sum(v * v for v in vector))
    if norm > 0:
        return [v / norm for v in vector]
    return vector


def cosine(a: list[float], b: list[float]) -> float:
    n = min(len(a), len(b))
    return sum(a[i] * b[i] for i in range(n))
