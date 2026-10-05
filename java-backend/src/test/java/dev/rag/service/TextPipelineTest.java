package dev.rag.service;

import org.junit.jupiter.api.Test;

import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.util.HexFormat;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class TextPipelineTest {
    @Test
    void rejectsInvalidChunkConfiguration() {
        assertThrows(IllegalArgumentException.class, () -> new TextPipeline(199, 0));
        assertThrows(IllegalArgumentException.class, () -> new TextPipeline(200, 200));
        assertThrows(IllegalArgumentException.class, () -> new TextPipeline(200, -1));
    }

    @Test
    void checksumMatchesSha256() throws Exception {
        var pipeline = new TextPipeline(200, 20);
        byte[] payload = "sample document".getBytes(StandardCharsets.UTF_8);
        String expected = HexFormat.of().formatHex(MessageDigest.getInstance("SHA-256").digest(payload));

        assertEquals(expected, pipeline.checksum(payload));
    }

    @Test
    void chunkTextsNormalizesWhitespaceAndReturnsNoChunksForBlankInput() {
        var pipeline = new TextPipeline(200, 20);

        assertEquals(List.of(), pipeline.chunkTexts(" \n\t "));
        assertEquals(List.of("alpha beta gamma"), pipeline.chunkTexts("  alpha\n\t beta   gamma  "));
    }

    @Test
    void chunkTextsSplitsAtWordsAndRetainsOverlap() {
        var pipeline = new TextPipeline(200, 20);
        String text = "alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima mike november oscar papa quebec romeo sierra tango uniform victor whiskey xray yankee zulu alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima mike november oscar papa quebec romeo sierra tango uniform victor whiskey xray yankee zulu";

        List<String> chunks = pipeline.chunkTexts(text);

        assertTrue(chunks.size() > 1);
        assertTrue(chunks.stream().allMatch(chunk -> chunk.length() <= 200));
        assertTrue(chunks.stream().allMatch(chunk -> chunk.equals(chunk.trim())));
        for (int i = 1; i < chunks.size(); i++) {
            var previousWords = new java.util.HashSet<>(java.util.Arrays.asList(chunks.get(i - 1).split(" ")));
            assertTrue(java.util.Arrays.stream(chunks.get(i).split(" ")).anyMatch(previousWords::contains));
        }
        assertTrue(chunks.getFirst().startsWith("alpha bravo charlie"));
    }

    @Test
    void localEmbeddingIsNormalizedAndCaseInsensitive() {
        var pipeline = new TextPipeline(200, 20);

        float[] vector = pipeline.embedLocal("RAG uses local embeddings");
        float[] sameVector = pipeline.embedLocal("rag uses local embeddings");
        double norm = 0;
        for (float value : vector) norm += value * value;

        assertEquals(1536, vector.length);
        assertEquals(1.0, norm, 1e-6);
        assertArrayEquals(vector, sameVector);
        assertEquals(0.0, pipeline.embedLocal("!!!")[0]);
    }
}
