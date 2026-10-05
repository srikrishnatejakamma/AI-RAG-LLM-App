package dev.rag.service;

import dev.rag.store.RagRepository;
import dev.rag.store.RagRepository.ChunkData;
import org.apache.tika.Tika;
import org.apache.pdfbox.Loader;
import org.apache.pdfbox.pdmodel.PDDocument;
import org.apache.pdfbox.text.PDFTextStripper;
import org.springframework.scheduling.annotation.Async;
import org.springframework.stereotype.Service;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.boot.context.event.ApplicationReadyEvent;
import org.springframework.context.event.EventListener;
import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.io.Writer;
import java.util.ArrayList;
import java.util.List;
import java.util.NoSuchElementException;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

@Service
public class IngestionService {
    private static final int MAX_EXTRACTED_CHARS = 2_000_000;
    private static final Logger log = LoggerFactory.getLogger(IngestionService.class);
    private final RagRepository repository;
    private final TextPipeline pipeline;
    private final EmbeddingService embeddings;
    private final AuditService audit;
    private final Tika tika = new Tika();
    @Autowired private OpenAIFileSearchService hostedFileSearch;
    @Value("${rag.retrieval.provider:local}") private String retrievalProvider;
    public IngestionService(RagRepository repository, TextPipeline pipeline, EmbeddingService embeddings, AuditService audit) {
        this.repository = repository; this.pipeline = pipeline; this.embeddings = embeddings; this.audit = audit;
        tika.setMaxStringLength(2_000_000);
    }
    @EventListener(ApplicationReadyEvent.class)
    public void recoverInterruptedJobs() { repository.recoverInterruptedIngestions(); }
    @Async("ingestionExecutor")
    public void process(String collectionId, String documentId, String extension, byte[] bytes, String actor, String requestId) {
        String hostedFileId = null;
        try {
            if (bytes == null || bytes.length == 0) {
                throw new IllegalArgumentException("No readable text was found in this document");
            }
            if ("openai-file-search".equalsIgnoreCase(retrievalProvider)) {
                var document = repository.document(collectionId, documentId);
                if (document.openaiFileId != null) hostedFileSearch.deleteFile(document.openaiFileId);
                hostedFileId = hostedFileSearch.uploadAndIndex(collectionId, documentId, document.name, bytes);
                repository.setOpenAiFileId(collectionId, documentId, hostedFileId);
                repository.complete(collectionId, documentId, List.of());
                audit.record(actor, "DOCUMENT_INGESTION_COMPLETED", "document", documentId, "READY", requestId);
                return;
            }
            var pages = "pdf".equals(extension) ? pdfPages(bytes) : List.of(new PageText(null, tika.parseToString(new ByteArrayInputStream(bytes))));
            var pieces = new ArrayList<PageText>();
            for (var page : pages) {
                if (page.text() == null || page.text().isBlank()) continue;
                for (String part : pipeline.chunkTexts(page.text())) pieces.add(new PageText(page.pageNumber(), part));
            }
            if (pieces.isEmpty()) throw new IllegalArgumentException("No readable text was found in this document");
            if (pieces.size() > 1800) throw new IllegalArgumentException("Document is too large after extraction; split it into smaller files");
            var vectors = embeddings.embedAll(pieces.stream().map(PageText::text).toList());
            var chunks = new ArrayList<ChunkData>(pieces.size());
            for (int i = 0; i < pieces.size(); i++) chunks.add(new ChunkData(pieces.get(i).text(), i + 1, pieces.get(i).pageNumber(), vectors.get(i)));
            repository.complete(collectionId, documentId, chunks);
            audit.record(actor, "DOCUMENT_INGESTION_COMPLETED", "document", documentId, "READY", requestId);
        } catch (Exception e) {
            if (hostedFileId != null && hostedFileSearch != null) {
                try { hostedFileSearch.deleteFile(hostedFileId); }
                catch (Exception cleanupFailure) { log.warn("Could not remove orphaned hosted file for document {}", documentId, cleanupFailure); }
            }
            String message = e instanceof IllegalArgumentException ? e.getMessage() : "Document processing failed. Check provider settings and retry.";
            log.warn("Ingestion failed for document {}", documentId, e);
            try { repository.fail(collectionId, documentId, message); }
            catch (NoSuchElementException deletedWhileProcessing) { /* Collection or document was removed during ingestion. */ }
            catch (Exception persistenceFailure) { log.error("Could not persist failed ingestion state for document {}", documentId, persistenceFailure); }
            audit.record(actor, "DOCUMENT_INGESTION_FAILED", "document", documentId, "FAILED", requestId);
        }
    }
    private List<PageText> pdfPages(byte[] bytes) throws IOException {
        try (PDDocument document = Loader.loadPDF(bytes)) {
            var stripper = new PDFTextStripper();
            stripper.setPageEnd("\f");
            var output = new BoundedWriter(MAX_EXTRACTED_CHARS);
            stripper.writeText(document, output);
            String[] pages = output.toString().split("\\f", -1);
            var result = new ArrayList<PageText>(pages.length);
            for (int i = 0; i < pages.length; i++) result.add(new PageText(i + 1, pages[i]));
            return result;
        }
    }
    private record PageText(Integer pageNumber, String text) {}
    private static final class BoundedWriter extends Writer {
        private final int maximum;
        private final StringBuilder content = new StringBuilder();
        BoundedWriter(int maximum) { this.maximum = maximum; }
        @Override public void write(char[] chars, int offset, int length) throws IOException {
            if (content.length() + length > maximum) throw new IOException("Extracted PDF text exceeds the configured limit");
            content.append(chars, offset, length);
        }
        @Override public void flush() {}
        @Override public void close() {}
        @Override public String toString() { return content.toString(); }
    }
}
