package dev.rag.model;

import java.time.Instant;
import java.util.List;

public final class Models {
    private Models() {}

    public record CollectionView(String id, String name, Instant createdAt, int documentCount) {}
    public record CreateCollectionRequest(String name) {}
    public record DocumentView(String id, String name, String status, int chunkCount, Instant uploadedAt, String error) {}
    public record Citation(String documentId, String documentName, int chunk, Integer page, String excerpt) {}
    public record ChatRequest(String question) {}
    public record ChatResponse(String answer, List<Citation> sources, boolean grounded) {}
    public record ErrorResponse(String error) {}
}
