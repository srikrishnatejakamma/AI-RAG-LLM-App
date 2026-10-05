package dev.rag.service;

import dev.rag.store.RagRepository;
import org.apache.pdfbox.pdmodel.PDDocument;
import org.apache.pdfbox.pdmodel.PDPage;
import org.junit.jupiter.api.Test;

import java.io.ByteArrayOutputStream;
import java.nio.charset.StandardCharsets;
import java.util.List;

import static org.mockito.ArgumentMatchers.anyList;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.doThrow;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

class IngestionServiceTest {
    private final RagRepository repository = mock(RagRepository.class);
    private final EmbeddingService embeddings = mock(EmbeddingService.class);
    private final AuditService audit = mock(AuditService.class);
    private final IngestionService ingestion = new IngestionService(
            repository, new TextPipeline(200, 20), embeddings, audit);

    @Test
    void ingestsReadableTextIntoNumberedChunksAndAuditsCompletion() throws Exception {
        when(embeddings.embedAll(anyList())).thenAnswer(invocation ->
                ((List<String>) invocation.getArgument(0)).stream().map(text -> new float[]{1f, 0f}).toList());
        String text = "This document contains enough readable information to create a useful indexed source passage.";

        ingestion.process("collection", "document", "txt", text.getBytes(StandardCharsets.UTF_8), "editor", "request");

        verify(repository).complete(eq("collection"), eq("document"), anyList());
        verify(audit).record("editor", "DOCUMENT_INGESTION_COMPLETED", "document", "document", "READY", "request");
    }

    @Test
    void failsWhenExtractionFindsNoReadableText() {
        ingestion.process("collection", "document", "txt", new byte[0], "editor", "request");

        verify(repository).fail("collection", "document", "No readable text was found in this document");
        verify(audit).record("editor", "DOCUMENT_INGESTION_FAILED", "document", "document", "FAILED", "request");
    }

    @Test
    void parsesPdfPagesAndFailsCleanlyWhenTheyContainNoText() throws Exception {
        byte[] pdf;
        try (var document = new PDDocument(); var output = new ByteArrayOutputStream()) {
            document.addPage(new PDPage());
            document.addPage(new PDPage());
            document.save(output);
            pdf = output.toByteArray();
        }

        ingestion.process("collection", "document", "pdf", pdf, "editor", "request");

        verify(repository).fail("collection", "document", "No readable text was found in this document");
        verify(audit).record("editor", "DOCUMENT_INGESTION_FAILED", "document", "document", "FAILED", "request");
    }

    @Test
    void convertsEmbeddingProviderErrorsToSafeFailureMessage() throws Exception {
        doThrow(new IllegalStateException("provider secret detail")).when(embeddings).embedAll(anyList());
        String text = "This document contains enough readable information to create a useful indexed source passage.";

        ingestion.process("collection", "document", "txt", text.getBytes(StandardCharsets.UTF_8), "editor", "request");

        verify(repository).fail(eq("collection"), eq("document"),
                eq("Document processing failed. Check provider settings and retry."));
        verify(audit).record("editor", "DOCUMENT_INGESTION_FAILED", "document", "document", "FAILED", "request");
    }
}
