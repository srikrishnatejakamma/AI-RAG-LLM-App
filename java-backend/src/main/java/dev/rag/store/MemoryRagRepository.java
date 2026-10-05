package dev.rag.store;

import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.stereotype.Component;
import java.time.Instant;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentLinkedDeque;
import java.util.concurrent.atomic.AtomicLong;
import java.util.stream.Collectors;

@Component
@ConditionalOnProperty(name = "rag.storage", havingValue = "memory", matchIfMissing = true)
public class MemoryRagRepository implements RagRepository {
    private static class StoredCollection {
        final String id = UUID.randomUUID().toString();
        final String name;
        final Instant createdAt = Instant.now();
        final Map<String, DocumentData> documents = new ConcurrentHashMap<>();
        StoredCollection(String name) { this.name = name; }
    }
    private final Map<String, StoredCollection> collections = new ConcurrentHashMap<>();
    private final ConcurrentLinkedDeque<AuditRecord> audit = new ConcurrentLinkedDeque<>();
    private final AtomicLong auditIds = new AtomicLong();
    public CollectionData create(String name) {
        var c = new StoredCollection(name); collections.put(c.id, c); return view(c);
    }
    public CollectionData get(String id) { var c = collections.get(id); return c == null ? null : view(c); }
    public CollectionData require(String id) {
        var c = collections.get(id);
        if (c == null) throw new NoSuchElementException("Collection not found");
        return view(c);
    }
    public void deleteCollection(String id) {
        if (collections.remove(id) == null) throw new NoSuchElementException("Collection not found");
    }
    public List<CollectionData> all() { return collections.values().stream().map(this::view).sorted(Comparator.comparing(c -> c.createdAt)).toList(); }
    public DocumentData findByChecksum(String collectionId, String checksum) {
        var c = requireRaw(collectionId);
        return c.documents.values().stream().filter(d -> d.checksum.equals(checksum)).findFirst().orElse(null);
    }
    public synchronized DocumentRegistration createPending(String collectionId, String name, String checksum) {
        var c = requireRaw(collectionId);
        var existing = findByChecksum(collectionId, checksum);
        if (existing != null) return new DocumentRegistration(existing, false);
        var d = new DocumentData(UUID.randomUUID().toString(), name, checksum, Instant.now(), "PROCESSING", null, List.of());
        c.documents.put(d.id, d); return new DocumentRegistration(d, true);
    }
    public boolean retryFailed(String collectionId, String documentId) {
        var d = requireDocument(collectionId, documentId);
        synchronized (d) {
            if (!"FAILED".equals(d.status)) return false;
            d.status = "PROCESSING"; d.error = null; return true;
        }
    }
    public void complete(String collectionId, String documentId, List<ChunkData> chunks) {
        var d = requireDocument(collectionId, documentId); d.chunks = List.copyOf(chunks); d.chunkCount = chunks.size(); d.status = "READY"; d.error = null;
    }
    public void fail(String collectionId, String documentId, String message) {
        var d = requireDocument(collectionId, documentId); d.status = "FAILED"; d.error = message;
    }
    public void recoverInterruptedIngestions() {}
    public boolean isAvailable() { return true; }
    public AuditRecord recordAudit(AuditRecord event) {
        var stored = new AuditRecord(auditIds.incrementAndGet(), event.occurredAt(), event.actor(), event.action(), event.resourceType(), event.resourceId(), event.outcome(), event.requestId());
        audit.addFirst(stored); while (audit.size() > 10_000) audit.pollLast(); return stored;
    }
    public List<AuditRecord> auditEvents(int limit, long beforeEventId) {
        return audit.stream().filter(e -> e.eventId() < beforeEventId).limit(limit).toList();
    }
    public List<DocumentSummary> documents(String collectionId) {
        return requireRaw(collectionId).documents.values().stream().sorted(Comparator.comparing(d -> d.uploadedAt))
            .map(d -> new DocumentSummary(d.id, d.name, d.status, d.chunkCount, d.uploadedAt, d.error)).toList();
    }
    public void deleteDocument(String collectionId, String documentId) {
        if (requireRaw(collectionId).documents.remove(documentId) == null) throw new NoSuchElementException("Document not found");
    }
    public DocumentData document(String collectionId, String documentId) { return requireDocument(collectionId, documentId); }
    public List<DocumentData> allDocuments(String collectionId) { return List.copyOf(requireRaw(collectionId).documents.values()); }
    public void setOpenAiFileId(String collectionId, String documentId, String fileId) { requireDocument(collectionId, documentId).openaiFileId = fileId; }
    public List<ScoredChunk> search(String collectionId, float[] query, int limit, double minimumScore) {
        return requireRaw(collectionId).documents.values().stream().filter(d -> "READY".equals(d.status))
            .flatMap(d -> d.chunks.stream().map(chunk -> new ScoredChunk(d, chunk, cosine(query, chunk.vector()))))
            .filter(match -> match.score() >= minimumScore)
            .sorted(Comparator.comparingDouble(ScoredChunk::score).reversed()).limit(limit).toList();
    }
    public List<ScoredChunk> summaryCandidates(String collectionId, int limit) {
        var documents = requireRaw(collectionId).documents.values().stream().filter(d -> "READY".equals(d.status))
            .filter(d -> !d.chunks.isEmpty()).sorted(Comparator.comparing(d -> d.uploadedAt)).toList();
        if (limit < 1 || documents.isEmpty()) return List.of();
        int perDocument = Math.max(1, limit / documents.size());
        var selected = new ArrayList<ScoredChunk>();
        for (var document : documents) {
            int count = Math.min(perDocument, document.chunks.size());
            for (int i = 0; i < count && selected.size() < limit; i++) {
                int index = count == 1 ? document.chunks.size() / 2 : (int) Math.round(i * (document.chunks.size() - 1.0) / (count - 1));
                selected.add(new ScoredChunk(document, document.chunks.get(index), 1.0));
            }
        }
        return List.copyOf(selected);
    }
    private StoredCollection requireRaw(String id) {
        var c = collections.get(id); if (c == null) throw new NoSuchElementException("Collection not found"); return c;
    }
    private DocumentData requireDocument(String collectionId, String documentId) {
        var d = requireRaw(collectionId).documents.get(documentId);
        if (d == null) throw new NoSuchElementException("Document not found"); return d;
    }
    private CollectionData view(StoredCollection c) { return new CollectionData(c.id, c.name, c.createdAt, c.documents.size()); }
    static double cosine(float[] a, float[] b) {
        double sum = 0; for (int i = 0; i < Math.min(a.length, b.length); i++) sum += a[i] * b[i]; return sum;
    }
}
