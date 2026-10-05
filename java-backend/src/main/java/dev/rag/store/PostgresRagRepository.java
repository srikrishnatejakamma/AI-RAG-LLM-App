package dev.rag.store;

import dev.rag.store.RagRepository.*;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.core.io.ClassPathResource;
import org.springframework.beans.factory.DisposableBean;
import org.springframework.jdbc.core.JdbcTemplate;
import com.zaxxer.hikari.HikariDataSource;
import org.springframework.jdbc.datasource.DataSourceTransactionManager;
import org.springframework.stereotype.Repository;
import org.springframework.transaction.support.TransactionTemplate;
import org.springframework.jdbc.datasource.init.ResourceDatabasePopulator;
import javax.sql.DataSource;
import java.sql.*;
import java.time.Instant;
import java.util.*;

@Repository
@ConditionalOnProperty(name = "rag.storage", havingValue = "postgres")
public class PostgresRagRepository implements RagRepository, DisposableBean {
    private final JdbcTemplate jdbc;
    private final TransactionTemplate transactions;
    private final HikariDataSource source;
    public PostgresRagRepository(@Value("${RAG_DATABASE_URL:jdbc:postgresql://localhost:5432/rag}") String url,
                                 @Value("${RAG_DATABASE_USER:rag}") String username,
                                 @Value("${RAG_DATABASE_PASSWORD:rag-local-only}") String password,
                                 @Value("${rag.embedding.provider:local}") String embeddingProvider,
                                 @Value("${rag.openai.embedding-model:text-embedding-3-small}") String embeddingModel,
                                 @Value("${rag.openai.base-url:https://api.openai.com/v1}") String embeddingBaseUrl) {
        source = new HikariDataSource();
        source.setJdbcUrl(url); source.setUsername(username); source.setPassword(password);
        source.setDriverClassName("org.postgresql.Driver");
        source.setMaximumPoolSize(12); source.setMinimumIdle(2); source.setConnectionTimeout(5000);
        source.setPoolName("rag-postgres");
        jdbc = new JdbcTemplate(source);
        jdbc.setFetchSize(100);
        jdbc.setQueryTimeout(20);
        transactions = new TransactionTemplate(new DataSourceTransactionManager(source));
        var initializer = new ResourceDatabasePopulator(new ClassPathResource("db/init.sql"));
        initializer.execute(source);
        String embeddingProfile = "openai".equalsIgnoreCase(embeddingProvider) ? "openai:" + embeddingBaseUrl + ":" + embeddingModel + ":1536" : "local:feature-hash-v1";
        jdbc.update("INSERT INTO system_settings(setting_key,setting_value) VALUES ('embedding_profile',?) ON CONFLICT(setting_key) DO NOTHING", embeddingProfile);
        String storedProfile = jdbc.queryForObject("SELECT setting_value FROM system_settings WHERE setting_key='embedding_profile'", String.class);
        if (!embeddingProfile.equals(storedProfile)) {
            Integer existingVectors = jdbc.queryForObject("SELECT count(*) FROM chunks", Integer.class);
            if (existingVectors != null && existingVectors > 0) {
                source.close();
                throw new IllegalStateException("The configured embedding model does not match stored vectors. Remove and re-upload documents before switching providers.");
            }
            jdbc.update("UPDATE system_settings SET setting_value=? WHERE setting_key='embedding_profile'", embeddingProfile);
        }
    }
    @Override public void destroy() { source.close(); }
    @Override public CollectionData create(String name) {
        String id = UUID.randomUUID().toString();
        jdbc.update("INSERT INTO collections(id,name) VALUES (?::uuid,?)", id, name);
        return get(id);
    }
    @Override public CollectionData get(String id) {
        return jdbc.query("SELECT c.id::text,c.name,c.created_at,count(d.id)::int AS documents FROM collections c LEFT JOIN documents d ON d.collection_id=c.id WHERE c.id=?::uuid GROUP BY c.id", rs -> rs.next() ? collection(rs) : null, id);
    }
    @Override public CollectionData require(String id) {
        var result = get(id); if (result == null) throw new NoSuchElementException("Collection not found"); return result;
    }
    @Override public void deleteCollection(String id) {
        require(id); jdbc.update("DELETE FROM collections WHERE id=?::uuid", id);
    }
    @Override public List<CollectionData> all() {
        return jdbc.query("SELECT c.id::text,c.name,c.created_at,count(d.id)::int AS documents FROM collections c LEFT JOIN documents d ON d.collection_id=c.id GROUP BY c.id ORDER BY c.created_at", (rs, row) -> collection(rs));
    }
    @Override public DocumentData findByChecksum(String collectionId, String checksum) {
        var result = jdbc.query("SELECT d.id::text,d.name,d.sha256,d.status,d.error_message,d.uploaded_at,(SELECT count(*) FROM chunks ch WHERE ch.document_id=d.id)::int AS chunks FROM documents d WHERE d.collection_id=?::uuid AND d.sha256=?", (rs, row) -> document(rs), collectionId, checksum);
        return result.isEmpty() ? null : result.getFirst();
    }
    @Override public DocumentRegistration createPending(String collectionId, String name, String checksum) {
        String id = UUID.randomUUID().toString();
        int inserted = jdbc.update("INSERT INTO documents(id,collection_id,name,sha256,status) VALUES (?::uuid,?::uuid,?,?,'PROCESSING') ON CONFLICT(collection_id,sha256) DO NOTHING", id, collectionId, name, checksum);
        if (inserted == 1) return new DocumentRegistration(new DocumentData(id, name, checksum, Instant.now(), "PROCESSING", null, List.of()), true);
        var existing = findByChecksum(collectionId, checksum);
        if (existing == null) throw new IllegalStateException("Document could not be registered");
        return new DocumentRegistration(existing, false);
    }
    @Override public boolean retryFailed(String collectionId, String documentId) {
        return jdbc.update("UPDATE documents SET status='PROCESSING',error_message=NULL WHERE id=?::uuid AND collection_id=?::uuid AND status='FAILED'", documentId, collectionId) == 1;
    }
    @Override public void complete(String collectionId, String documentId, List<ChunkData> chunks) {
        transactions.executeWithoutResult(status -> {
            jdbc.update("DELETE FROM chunks WHERE collection_id=?::uuid AND document_id=?::uuid", collectionId, documentId);
            jdbc.batchUpdate("INSERT INTO chunks(collection_id,document_id,chunk_index,page_number,content,embedding) VALUES (?::uuid,?::uuid,?,?,?,?::vector)", chunks, 100,
                (statement, chunk) -> {
                    statement.setString(1, collectionId); statement.setString(2, documentId); statement.setInt(3, chunk.index());
                    if (chunk.pageNumber() == null) statement.setNull(4, Types.INTEGER); else statement.setInt(4, chunk.pageNumber());
                    statement.setString(5, chunk.text()); statement.setString(6, vectorLiteral(chunk.vector()));
                });
            jdbc.update("UPDATE documents SET status='READY',error_message=NULL WHERE id=?::uuid AND collection_id=?::uuid", documentId, collectionId);
        });
    }
    @Override public void fail(String collectionId, String documentId, String message) {
        jdbc.update("UPDATE documents SET status='FAILED',error_message=? WHERE id=?::uuid AND collection_id=?::uuid", message, documentId, collectionId);
    }
    @Override public void recoverInterruptedIngestions() {
        jdbc.update("UPDATE documents SET status='FAILED',error_message='Processing was interrupted by a server restart. Upload the same file to retry.' WHERE status='PROCESSING'");
    }
    @Override public boolean isAvailable() { return Integer.valueOf(1).equals(jdbc.queryForObject("SELECT 1", Integer.class)); }
    @Override public AuditRecord recordAudit(AuditRecord event) {
        Long id = jdbc.queryForObject("INSERT INTO audit_events(occurred_at,actor,action,resource_type,resource_id,outcome,request_id) VALUES (?,?,?,?,?,?,?) RETURNING event_id",
            Long.class, Timestamp.from(event.occurredAt()), event.actor(), event.action(), event.resourceType(), event.resourceId(), event.outcome(), event.requestId());
        return new AuditRecord(id, event.occurredAt(), event.actor(), event.action(), event.resourceType(), event.resourceId(), event.outcome(), event.requestId());
    }
    @Override public List<AuditRecord> auditEvents(int limit, long beforeEventId) {
        return jdbc.query("SELECT event_id,occurred_at,actor,action,resource_type,resource_id,outcome,request_id FROM audit_events WHERE event_id<? ORDER BY event_id DESC LIMIT ?",
            (rs, row) -> new AuditRecord(rs.getLong("event_id"), instant(rs.getTimestamp("occurred_at")), rs.getString("actor"), rs.getString("action"), rs.getString("resource_type"), rs.getString("resource_id"), rs.getString("outcome"), rs.getString("request_id")), beforeEventId, limit);
    }
    @Override public List<DocumentSummary> documents(String collectionId) {
        require(collectionId);
        return jdbc.query("SELECT d.id::text,d.name,d.status,d.uploaded_at,d.error_message,count(ch.id)::int AS chunks FROM documents d LEFT JOIN chunks ch ON ch.document_id=d.id WHERE d.collection_id=?::uuid GROUP BY d.id ORDER BY d.uploaded_at", (rs, row) -> new DocumentSummary(rs.getString("id"), rs.getString("name"), rs.getString("status"), rs.getInt("chunks"), instant(rs.getTimestamp("uploaded_at")), rs.getString("error_message")), collectionId);
    }
    @Override public void deleteDocument(String collectionId, String documentId) {
        int count = jdbc.update("DELETE FROM documents WHERE id=?::uuid AND collection_id=?::uuid", documentId, collectionId);
        if (count == 0) throw new NoSuchElementException("Document not found");
    }
    @Override public DocumentData document(String collectionId, String documentId) {
        var rows = jdbc.query("SELECT d.id::text,d.name,d.sha256,d.status,d.error_message,d.uploaded_at,d.openai_file_id,(SELECT count(*) FROM chunks ch WHERE ch.document_id=d.id)::int AS chunks FROM documents d WHERE d.id=?::uuid AND d.collection_id=?::uuid", (rs, row) -> document(rs), documentId, collectionId);
        if (rows.isEmpty()) throw new NoSuchElementException("Document not found");
        return rows.getFirst();
    }
    @Override public List<DocumentData> allDocuments(String collectionId) {
        return jdbc.query("SELECT d.id::text,d.name,d.sha256,d.status,d.error_message,d.uploaded_at,d.openai_file_id,(SELECT count(*) FROM chunks ch WHERE ch.document_id=d.id)::int AS chunks FROM documents d WHERE d.collection_id=?::uuid", (rs, row) -> document(rs), collectionId);
    }
    @Override public void setOpenAiFileId(String collectionId, String documentId, String fileId) {
        int changed = jdbc.update("UPDATE documents SET openai_file_id=? WHERE id=?::uuid AND collection_id=?::uuid", fileId, documentId, collectionId);
        if (changed == 0) throw new NoSuchElementException("Document not found");
    }
    @Override public List<ScoredChunk> search(String collectionId, float[] query, int limit, double minimumScore) {
        String vector = vectorLiteral(query);
        return jdbc.query("SELECT d.id::text AS doc_id,d.name,d.sha256,d.status,d.error_message,d.uploaded_at,c.chunk_index,c.page_number,c.content,1-(c.embedding <=> ?::vector) AS score FROM chunks c JOIN documents d ON d.id=c.document_id WHERE c.collection_id=?::uuid AND d.status='READY' AND 1-(c.embedding <=> ?::vector)>=? ORDER BY c.embedding <=> ?::vector LIMIT ?", (rs, row) -> {
            var document = new DocumentData(rs.getString("doc_id"), rs.getString("name"), rs.getString("sha256"), instant(rs.getTimestamp("uploaded_at")), rs.getString("status"), rs.getString("error_message"), List.of());
            int page = rs.getInt("page_number"); Integer pageNumber = rs.wasNull() ? null : page;
            var chunk = new ChunkData(rs.getString("content"), rs.getInt("chunk_index"), pageNumber, new float[0]);
            return new ScoredChunk(document, chunk, rs.getDouble("score"));
        }, vector, collectionId, vector, minimumScore, vector, limit);
    }
    @Override public List<ScoredChunk> summaryCandidates(String collectionId, int limit) {
        if (limit < 1) return List.of();
        String sql = "WITH ranked AS (" +
            "SELECT d.id::text AS doc_id,d.name,d.sha256,d.status,d.error_message,d.uploaded_at,c.chunk_index,c.page_number,c.content," +
            "row_number() OVER(PARTITION BY d.id ORDER BY c.chunk_index) AS rn,count(*) OVER(PARTITION BY d.id) AS per_document," +
            "(SELECT count(*) FROM documents ready WHERE ready.collection_id=?::uuid AND ready.status='READY') AS document_count " +
            "FROM chunks c JOIN documents d ON d.id=c.document_id WHERE c.collection_id=?::uuid AND d.status='READY'" +
            "), sampled AS (SELECT *,GREATEST(1,CEIL(per_document::numeric/GREATEST(1,FLOOR(?::numeric/GREATEST(1,document_count)))))::int AS stride FROM ranked) " +
            "SELECT * FROM sampled WHERE MOD(rn-1,stride)=0 ORDER BY uploaded_at,chunk_index LIMIT ?";
        return jdbc.query(sql, (rs, row) -> {
            var document = new DocumentData(rs.getString("doc_id"), rs.getString("name"), rs.getString("sha256"), instant(rs.getTimestamp("uploaded_at")), rs.getString("status"), rs.getString("error_message"), List.of());
            int page = rs.getInt("page_number"); Integer pageNumber = rs.wasNull() ? null : page;
            return new ScoredChunk(document, new ChunkData(rs.getString("content"), rs.getInt("chunk_index"), pageNumber, new float[0]), 1.0);
        }, collectionId, collectionId, limit, limit);
    }
    private CollectionData collection(ResultSet rs) throws SQLException { return new CollectionData(rs.getString("id"), rs.getString("name"), instant(rs.getTimestamp("created_at")), rs.getInt("documents")); }
    private DocumentData document(ResultSet rs) throws SQLException {
        var document = new DocumentData(rs.getString("id"), rs.getString("name"), rs.getString("sha256"), instant(rs.getTimestamp("uploaded_at")), rs.getString("status"), rs.getString("error_message"), List.of());
        document.chunkCount = rs.getInt("chunks");
        try { document.openaiFileId = rs.getString("openai_file_id"); } catch (SQLException ignored) { /* query does not include hosted ID */ }
        return document;
    }
    private static Instant instant(Timestamp ts) { return ts.toInstant(); }
    private static String vectorLiteral(float[] values) {
        var out = new StringBuilder("[");
        for (int i = 0; i < values.length; i++) { if (i > 0) out.append(','); out.append(values[i]); }
        return out.append(']').toString();
    }
}
