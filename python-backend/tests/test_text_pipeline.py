import hashlib
import unittest

from app.text_pipeline import chunk_text, checksum, cosine, embed_local, normalize_text


class TextPipelineTests(unittest.TestCase):
    def test_normalize_text_collapses_whitespace_and_trims(self):
        self.assertEqual(normalize_text("  alpha\n\t beta   gamma  "), "alpha beta gamma")

    def test_chunk_text_returns_empty_for_blank_input(self):
        self.assertEqual(chunk_text(" \n\t ", chunk_size=20, overlap=4), [])

    def test_chunk_text_normalizes_and_overlaps_at_word_boundaries(self):
        text = "alpha bravo charlie delta echo foxtrot golf hotel india juliet"

        chunks = chunk_text(text, chunk_size=20, overlap=5)

        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(len(chunk) <= 20 for chunk in chunks))
        self.assertTrue(all(chunk == chunk.strip() for chunk in chunks))
        self.assertTrue(all(chunk in text for chunk in chunks))
        for left, right in zip(chunks, chunks[1:]):
            self.assertTrue(set(left.split()).intersection(right.split()))

    def test_checksum_matches_sha256(self):
        payload = b"sample document"
        self.assertEqual(checksum(payload), hashlib.sha256(payload).hexdigest())

    def test_local_embedding_is_normalized_and_case_insensitive(self):
        vector = embed_local("RAG uses local embeddings")
        same_vector = embed_local("rag uses local embeddings")

        self.assertEqual(len(vector), 1536)
        self.assertAlmostEqual(sum(value * value for value in vector), 1.0)
        self.assertEqual(vector, same_vector)

    def test_empty_embedding_is_zero_vector(self):
        self.assertEqual(embed_local("!!!", dimensions=8), [0.0] * 8)

    def test_cosine_uses_shared_dimensions(self):
        self.assertAlmostEqual(cosine([1.0, 2.0], [3.0, 4.0, 100.0]), 11.0)


if __name__ == "__main__":
    unittest.main()
