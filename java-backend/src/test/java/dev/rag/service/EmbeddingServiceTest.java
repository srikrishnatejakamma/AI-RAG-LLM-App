package dev.rag.service;

import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class EmbeddingServiceTest {
    private final TextPipeline pipeline = new TextPipeline(200, 20);

    @Test
    void localProviderEmbedsOneOrManyTextsAndHandlesEmptyInput() {
        var service = new EmbeddingService(pipeline, "", "text-embedding-3-small", "local", "http://localhost");

        assertEquals("local", service.providerName());
        assertEquals(1536, service.embed("policy").length);
        assertEquals(2, service.embedAll(List.of("first", "second")).size());
        assertEquals(List.of(), service.embedAll(List.of()));
    }

    @Test
    void openAiProviderRequiresApiKeyBeforeMakingARequest() {
        var service = new EmbeddingService(pipeline, " ", "text-embedding-3-small", "openai", "http://localhost");

        assertEquals("openai", service.providerName());
        assertThrows(IllegalStateException.class, () -> service.embed("policy"));
    }

    @Test
    void rejectsUnsupportedProviderConfiguration() {
        assertThrows(IllegalArgumentException.class,
                () -> new EmbeddingService(pipeline, "", "model", "other", "http://localhost"));
    }
}
