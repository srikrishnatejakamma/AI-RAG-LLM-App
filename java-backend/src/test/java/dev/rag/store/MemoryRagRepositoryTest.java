package dev.rag.store;

import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

class MemoryRagRepositoryTest {
    @Test
    void documentLifecycleDeduplicatesRetriesAndCompletes() {
        var repository = new MemoryRagRepository();
        var collection = repository.create("Research");

        var registration = repository.createPending(collection.id, "notes.txt", "sha256");
        var duplicate = repository.createPending(collection.id, "notes.txt", "sha256");
        assertTrue(registration.created());
        assertFalse(duplicate.created());
        assertEquals(registration.document().id, duplicate.document().id);

        repository.fail(collection.id, registration.document().id, "temporary error");
        assertEquals("FAILED", registration.document().status);
        assertTrue(repository.retryFailed(collection.id, registration.document().id));
        assertEquals("PROCESSING", registration.document().status);
        assertNull(registration.document().error);

        var chunk = new RagRepository.ChunkData("Relevant source text", 1, new float[]{1f, 0f});
        repository.complete(collection.id, registration.document().id, List.of(chunk));

        assertEquals("READY", registration.document().status);
        assertEquals(1, registration.document().chunkCount);
        assertEquals(1, repository.documents(collection.id).size());
    }

    @Test
    void searchReturnsReadyChunksAboveTheScoreThresholdInDescendingOrder() {
        var repository = new MemoryRagRepository();
        var collection = repository.create("Research");
        var document = repository.createPending(collection.id, "notes.txt", "sha256").document();
        repository.complete(collection.id, document.id, List.of(
                new RagRepository.ChunkData("Best match", 1, new float[]{1f, 0f}),
                new RagRepository.ChunkData("Weaker match", 2, new float[]{0.5f, 0.5f})
        ));

        var matches = repository.search(collection.id, new float[]{1f, 0f}, 10, 0.4);

        assertEquals(2, matches.size());
        assertEquals("Best match", matches.get(0).chunk().text());
        assertEquals(1.0, matches.get(0).score());
        assertEquals(0.5, matches.get(1).score());
        assertNotNull(matches.get(0).document());
    }

    @Test
    void searchExcludesProcessingDocumentsAndHonorsLimitAndMinimumScore() {
        var repository = new MemoryRagRepository();
        var collection = repository.create("Research");
        var processing = repository.createPending(collection.id, "pending.txt", "pending").document();
        var ready = repository.createPending(collection.id, "ready.txt", "ready").document();
        repository.complete(collection.id, ready.id, List.of(
                new RagRepository.ChunkData("Strong", 1, new float[]{1f, 0f}),
                new RagRepository.ChunkData("Weak", 2, new float[]{0.2f, 0.8f})
        ));

        var matches = repository.search(collection.id, new float[]{1f, 0f}, 1, 0.5);

        assertEquals("PROCESSING", processing.status);
        assertEquals(1, matches.size());
        assertEquals("Strong", matches.getFirst().chunk().text());
    }

    @Test
    void summaryCandidatesSampleChunksAcrossReadyDocuments() {
        var repository = new MemoryRagRepository();
        var collection = repository.create("Handbook");
        var first = repository.createPending(collection.id, "first.txt", "first").document();
        var second = repository.createPending(collection.id, "second.txt", "second").document();
        repository.complete(collection.id, first.id, List.of(
                new RagRepository.ChunkData("first opening", 1, new float[]{1f}),
                new RagRepository.ChunkData("first middle", 2, new float[]{1f}),
                new RagRepository.ChunkData("first ending", 3, new float[]{1f})
        ));
        repository.complete(collection.id, second.id, List.of(
                new RagRepository.ChunkData("second opening", 1, new float[]{1f}),
                new RagRepository.ChunkData("second ending", 2, new float[]{1f})
        ));

        var candidates = repository.summaryCandidates(collection.id, 4);

        assertEquals(4, candidates.size());
        assertTrue(candidates.stream().anyMatch(candidate -> candidate.document().id.equals(first.id)));
        assertTrue(candidates.stream().anyMatch(candidate -> candidate.document().id.equals(second.id)));
        assertEquals(1.0, candidates.getFirst().score());
    }

    @Test
    void summaryCandidatesCanReturnEveryChunkForDocumentWideSummaries() {
        var repository = new MemoryRagRepository();
        var collection = repository.create("Handbook");
        var document = repository.createPending(collection.id, "handbook.txt", "handbook").document();
        repository.complete(collection.id, document.id, List.of(
                new RagRepository.ChunkData("first", 1, new float[]{1f}),
                new RagRepository.ChunkData("second", 2, new float[]{1f}),
                new RagRepository.ChunkData("third", 3, new float[]{1f})
        ));

        var candidates = repository.summaryCandidates(collection.id, Integer.MAX_VALUE);

        assertEquals(List.of("first", "second", "third"), candidates.stream().map(match -> match.chunk().text()).toList());
    }

    @Test
    void missingCollectionAndDocumentOperationsFailClearly() {
        var repository = new MemoryRagRepository();
        assertNull(repository.get("missing"));
        org.junit.jupiter.api.Assertions.assertThrows(java.util.NoSuchElementException.class, () -> repository.require("missing"));
        org.junit.jupiter.api.Assertions.assertThrows(java.util.NoSuchElementException.class, () -> repository.deleteCollection("missing"));

        var collection = repository.create("Research");
        org.junit.jupiter.api.Assertions.assertThrows(java.util.NoSuchElementException.class,
                () -> repository.deleteDocument(collection.id, "missing"));
    }

    @Test
    void auditEventsReceiveIncreasingIdsAndCanBePaged() {
        var repository = new MemoryRagRepository();
        var first = new RagRepository.AuditRecord(0, java.time.Instant.now(), "admin", "CREATE", "collection", "one", "SUCCESS", "r1");
        var second = new RagRepository.AuditRecord(0, java.time.Instant.now(), "admin", "DELETE", "collection", "one", "SUCCESS", "r2");

        var savedFirst = repository.recordAudit(first);
        var savedSecond = repository.recordAudit(second);

        assertEquals(1, savedFirst.eventId());
        assertEquals(2, savedSecond.eventId());
        assertEquals(List.of(savedSecond), repository.auditEvents(1, Long.MAX_VALUE));
        assertEquals(List.of(savedFirst), repository.auditEvents(10, savedSecond.eventId()));
    }
}
