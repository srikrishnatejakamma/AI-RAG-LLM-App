package dev.rag.api;

import dev.rag.model.Models.ChatRequest;
import dev.rag.model.Models.CreateCollectionRequest;
import dev.rag.service.AnswerProviderReadinessService;
import dev.rag.service.AuditService;
import dev.rag.service.AuthReadinessService;
import dev.rag.service.EmbeddingService;
import dev.rag.service.GroundedAnswerService;
import dev.rag.service.IngestionService;
import dev.rag.service.TextPipeline;
import dev.rag.store.MemoryRagRepository;
import dev.rag.store.RagRepository;
import dev.rag.store.RagRepository.ChunkData;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.springframework.mock.web.MockMultipartFile;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.Mockito.doThrow;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

class RagControllerTest {
    private MemoryRagRepository repository;
    private IngestionService ingestion;
    private GroundedAnswerService answerService;
    private RagController controller;

    @BeforeEach
    void setUp() {
        repository = new MemoryRagRepository();
        ingestion = mock(IngestionService.class);
        answerService = mock(GroundedAnswerService.class);
        var answerReadiness = mock(AnswerProviderReadinessService.class);
        when(answerReadiness.asHealthFields()).thenReturn(new AnswerProviderReadinessService("auto", "").asHealthFields());
        var authReadiness = new AuthReadinessService();
        controller = new RagController(
                repository,
                new TextPipeline(200, 20),
                new EmbeddingService(new TextPipeline(200, 20), "", "model", "local", "http://localhost"),
                answerService,
                answerReadiness,
                authReadiness,
                ingestion,
                mock(AuditService.class),
                1024,
                0.04,
                10,
                "memory");
    }

    @AfterEach
    void clearSecurityContext() {
        org.springframework.security.core.context.SecurityContextHolder.clearContext();
    }

    @Test
    void createCollectionTrimsNameAndRejectsBlankOrLongNames() {
        var created = controller.create(new CreateCollectionRequest("  Research  "));

        assertEquals("Research", created.name());
        assertEquals(0, created.documentCount());
        assertThrows(IllegalArgumentException.class, () -> controller.create(new CreateCollectionRequest("  ")));
        assertThrows(IllegalArgumentException.class, () -> controller.create(new CreateCollectionRequest("x".repeat(81))));
        assertThrows(IllegalArgumentException.class, () -> controller.create(null));
    }

    @Test
    void healthReportsStorageAndProviderReadiness() {
        var response = controller.health();

        assertEquals(200, response.getStatusCode().value());
        assertEquals("ok", response.getBody().get("status"));
        assertEquals("memory", response.getBody().get("storage"));
        assertEquals("local", response.getBody().get("embeddingProvider"));
        assertEquals("degraded", response.getBody().get("answerProviderStatus"));
    }

    @Test
    void healthReturnsServiceUnavailableWhenStorageProbeFails() {
        RagRepository unavailable = mock(RagRepository.class);
        when(unavailable.isAvailable()).thenThrow(new IllegalStateException("database detail"));
        var readiness = mock(AnswerProviderReadinessService.class);
        when(readiness.asHealthFields()).thenReturn(new AnswerProviderReadinessService("auto", "key").asHealthFields());
        var degradedController = new RagController(
                unavailable,
                new TextPipeline(200, 20),
                new EmbeddingService(new TextPipeline(200, 20), "", "model", "local", "http://localhost"),
                answerService,
                readiness,
                new AuthReadinessService(),
                ingestion,
                mock(AuditService.class),
                1024,
                0.04,
                10,
                "postgres");

        var response = degradedController.health();

        assertEquals(503, response.getStatusCode().value());
        assertEquals("degraded", response.getBody().get("status"));
        assertEquals("postgres", response.getBody().get("storage"));
        assertEquals("ok", response.getBody().get("answerProviderStatus"));
    }

    @Test
    void collectionAndDocumentEndpointsValidateIdsAndMissingResources() {
        assertThrows(IllegalArgumentException.class, () -> controller.documents("not-a-uuid"));
        assertThrows(IllegalArgumentException.class, () -> controller.deleteCollection("not-a-uuid"));
        assertThrows(IllegalArgumentException.class, () -> controller.deleteDocument("not-a-uuid", "also-invalid"));
        assertThrows(java.util.NoSuchElementException.class,
                () -> controller.documents("00000000-0000-0000-0000-000000000001"));

        var collection = repository.create("Research");
        assertEquals(0, controller.documents(collection.id).size());
        assertThrows(java.util.NoSuchElementException.class,
                () -> controller.deleteDocument(collection.id, "00000000-0000-0000-0000-000000000002"));
        assertEquals(204, controller.deleteCollection(collection.id).getStatusCode().value());
    }

    @Test
    void uploadValidatesEmptySizeExtensionAndDetectedContent() throws Exception {
        var collection = repository.create("Research");

        assertThrows(IllegalArgumentException.class,
                () -> controller.upload(collection.id, new MockMultipartFile("file", "empty.txt", "text/plain", new byte[0])));
        assertThrows(IllegalArgumentException.class,
                () -> controller.upload(collection.id, new MockMultipartFile("file", "large.txt", "text/plain", new byte[1025])));
        assertThrows(IllegalArgumentException.class,
                () -> controller.upload(collection.id, new MockMultipartFile("file", "image.png", "image/png", "data".getBytes())));
        assertThrows(IllegalArgumentException.class,
                () -> controller.upload(collection.id, new MockMultipartFile("file", "notes.txt", "text/plain", "%PDF-1.4 invalid".getBytes())));
        assertThrows(IllegalArgumentException.class,
                () -> controller.upload("not-a-uuid", new MockMultipartFile("file", "notes.txt", "text/plain", "text".getBytes())));
    }

    @Test
    void uploadSanitizesNameQueuesIngestionAndDeduplicatesContent() throws Exception {
        var collection = repository.create("Research");
        var file = new MockMultipartFile("file", "../../notes.txt", "text/plain",
                "A readable document with multiple useful words.".getBytes());

        var accepted = controller.upload(collection.id, file);
        var duplicate = controller.upload(collection.id, file);

        assertEquals(202, accepted.getStatusCode().value());
        assertEquals("notes.txt", accepted.getBody().name());
        assertEquals("PROCESSING", accepted.getBody().status());
        assertEquals(200, duplicate.getStatusCode().value());
        assertEquals(accepted.getBody().id(), duplicate.getBody().id());
        assertEquals(1, repository.documents(collection.id).size());
    }

    @Test
    void uploadRetriesPreviouslyFailedDocumentAndReportsQueueRejection() throws Exception {
        var collection = repository.create("Research");
        byte[] bytes = "A readable document with multiple useful words.".getBytes();
        var first = controller.upload(collection.id, new MockMultipartFile("file", "notes.txt", "text/plain", bytes));
        repository.fail(collection.id, first.getBody().id(), "prior failure");

        var retried = controller.upload(collection.id, new MockMultipartFile("file", "notes.txt", "text/plain", bytes));
        assertEquals(202, retried.getStatusCode().value());
        assertEquals("PROCESSING", retried.getBody().status());

        var rejectedCollection = repository.create("Another");
        doThrow(new IllegalStateException("queue full")).when(ingestion).process(
                org.mockito.ArgumentMatchers.eq(rejectedCollection.id),
                org.mockito.ArgumentMatchers.anyString(),
                org.mockito.ArgumentMatchers.eq("txt"),
                org.mockito.ArgumentMatchers.any(byte[].class),
                org.mockito.ArgumentMatchers.anyString(),
                org.mockito.ArgumentMatchers.nullable(String.class));
        assertThrows(IllegalStateException.class, () -> controller.upload(rejectedCollection.id,
                new MockMultipartFile("file", "queued.txt", "text/plain", "another readable document".getBytes())));
        assertEquals("FAILED", repository.documents(rejectedCollection.id).getFirst().status());
    }

    @Test
    void chatValidatesQuestionAndReportsNoEvidenceOrProcessing() {
        var collection = repository.create("Research");
        assertThrows(IllegalArgumentException.class, () -> controller.chat("not-a-uuid", new ChatRequest("question")));
        assertThrows(IllegalArgumentException.class, () -> controller.chat(collection.id, new ChatRequest("  ")));
        assertThrows(IllegalArgumentException.class, () -> controller.chat(collection.id, new ChatRequest("x".repeat(2001))));
        assertThrows(java.util.NoSuchElementException.class, () -> controller.chat(
                "00000000-0000-0000-0000-000000000001", new ChatRequest("question")));

        var empty = controller.chat(collection.id, new ChatRequest("question"));
        assertTrue(!empty.grounded());
        assertTrue(empty.answer().contains("couldn't find enough information"));

        var pending = repository.createPending(collection.id, "pending.txt", "pending-sum").document();
        var processing = controller.chat(collection.id, new ChatRequest("question"));
        assertTrue(!processing.grounded());
        assertTrue(processing.answer().contains("still being processed"));
    }

    @Test
    void chatDoesNotFillResultsWithChunksBelowSimilarityThreshold() {
        var collection = repository.create("Research");
        var document = repository.createPending(collection.id, "unrelated.txt", "unrelated-sum").document();
        var embeddings = new EmbeddingService(new TextPipeline(200, 20), "", "model", "local", "http://localhost");
        float[] queryVector = embeddings.embed("retention policy");
        float[] unrelatedVector = queryVector.clone();
        for (int i = 0; i < unrelatedVector.length; i++) unrelatedVector[i] = -unrelatedVector[i];
        repository.complete(collection.id, document.id,
                List.of(new ChunkData("The office is near the river and has a garden.", 1, unrelatedVector)));

        var response = controller.chat(collection.id, new ChatRequest("retention policy"));

        assertTrue(!response.grounded());
        assertTrue(response.sources().isEmpty());
    }

    @Test
    void chatReturnsGroundedAnswerWithCitationWhenAReadyChunkMatches() {
        var collection = repository.create("Research");
        var document = repository.createPending(collection.id, "policy.txt", "policy-sum").document();
        String text = "The retention policy requires teams to review records every year. " + "Evidence detail. ".repeat(20);
        repository.complete(collection.id, document.id,
                List.of(new ChunkData(text, 1, 7, new EmbeddingService(new TextPipeline(200, 20), "", "model", "local", "http://localhost").embed("retention policy"))));
        when(answerService.answer(org.mockito.ArgumentMatchers.eq("retention policy"), org.mockito.ArgumentMatchers.anyList()))
                .thenReturn("The policy requires annual review.");

        var response = controller.chat(collection.id, new ChatRequest("retention policy"));

        assertTrue(response.grounded());
        assertEquals("The policy requires annual review.", response.answer());
        assertEquals(1, response.sources().size());
        assertEquals(7, response.sources().getFirst().page());
        assertTrue(response.sources().getFirst().excerpt().length() <= 220);
    }
}
