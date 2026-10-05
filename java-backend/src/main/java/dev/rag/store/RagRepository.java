package dev.rag.store;

import java.time.Instant;
import java.util.List;
import java.util.NoSuchElementException;

public interface RagRepository {
    record ChunkData(String text, int index, Integer pageNumber, float[] vector) {
        public ChunkData(String text, int index, float[] vector) { this(text, index, null, vector); }
    }
    record ScoredChunk(DocumentData document, ChunkData chunk, double score) {}
    record DocumentSummary(String id, String name, String status, int chunkCount, Instant uploadedAt, String error) {}
    record DocumentRegistration(DocumentData document, boolean created) {}
    record AuditRecord(long eventId, Instant occurredAt, String actor, String action, String resourceType,
                       String resourceId, String outcome, String requestId) {}

    class CollectionData {
        public final String id;
        public final String name;
        public final Instant createdAt;
        public final int documentCount;
        public CollectionData(String id, String name, Instant createdAt, int documentCount) {
            this.id = id; this.name = name; this.createdAt = createdAt; this.documentCount = documentCount;
        }
    }
    class DocumentData {
        public final String id, name, checksum;
        public final Instant uploadedAt;
        public volatile String status;
        public volatile String error;
        public volatile List<ChunkData> chunks;
        public volatile int chunkCount;
        public volatile String openaiFileId;
        public DocumentData(String id, String name, String checksum, Instant uploadedAt, String status, String error, List<ChunkData> chunks) {
            this.id = id; this.name = name; this.checksum = checksum; this.uploadedAt = uploadedAt;
            this.status = status; this.error = error; this.chunks = List.copyOf(chunks);
            this.chunkCount = chunks.size();
        }
    }

    CollectionData create(String name);
    CollectionData get(String id);
    CollectionData require(String id) throws NoSuchElementException;
    void deleteCollection(String id);
    List<CollectionData> all();
    DocumentData findByChecksum(String collectionId, String checksum);
    DocumentRegistration createPending(String collectionId, String name, String checksum);
    boolean retryFailed(String collectionId, String documentId);
    void complete(String collectionId, String documentId, List<ChunkData> chunks);
    void fail(String collectionId, String documentId, String message);
    void recoverInterruptedIngestions();
    boolean isAvailable();
    AuditRecord recordAudit(AuditRecord event);
    List<AuditRecord> auditEvents(int limit, long beforeEventId);
    List<DocumentSummary> documents(String collectionId);
    void deleteDocument(String collectionId, String documentId);
    DocumentData document(String collectionId, String documentId);
    List<DocumentData> allDocuments(String collectionId);
    void setOpenAiFileId(String collectionId, String documentId, String fileId);
    List<ScoredChunk> search(String collectionId, float[] query, int limit, double minimumScore);
    List<ScoredChunk> summaryCandidates(String collectionId, int limit);
}
