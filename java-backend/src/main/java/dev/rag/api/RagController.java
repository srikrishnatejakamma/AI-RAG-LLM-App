package dev.rag.api;

import dev.rag.model.Models.*;
import dev.rag.service.*;
import dev.rag.store.RagRepository;
import dev.rag.store.RagRepository.*;
import org.slf4j.MDC;
import org.springframework.security.core.Authentication;
import org.springframework.security.core.context.SecurityContextHolder;
import org.springframework.security.access.prepost.PreAuthorize;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.security.SecurityRequirement;
import io.swagger.v3.oas.annotations.tags.Tag;
import org.springframework.security.web.csrf.CsrfToken;
import org.apache.tika.Tika;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.*;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;
import java.util.*;

@RestController
@RequestMapping("/api")
@CrossOrigin(origins = {"http://localhost:5173", "http://127.0.0.1:5173"})
@Tag(name = "RAG", description = "Manage collections, ingest documents, and ask grounded questions")
public class RagController {
    private static final Set<String> ALLOWED = Set.of("pdf", "docx", "txt");
    private final RagRepository repository;
    private final TextPipeline pipeline;
    private final EmbeddingService embeddings;
    private final GroundedAnswerService answerService;
    private final AnswerProviderReadinessService answerProviderReadiness;
    private final AuthReadinessService authReadiness;
    private final IngestionService ingestion;
    private final AuditService audit;
    private final long maxBytes;
    private final double minimumScore;
    private final int retrievalTopK;
    private final String storage;
    @Autowired private OpenAIFileSearchService hostedFileSearch;
    @Value("${rag.retrieval.provider:local}") private String retrievalProvider;
    public RagController(RagRepository repository, TextPipeline pipeline, EmbeddingService embeddings,
                         GroundedAnswerService answerService, AnswerProviderReadinessService answerProviderReadiness,
                         AuthReadinessService authReadiness,
                         IngestionService ingestion, AuditService audit,
                         @Value("${rag.max-upload-bytes:10485760}") long maxBytes,
                         @Value("${rag.retrieval.minimum-score:0.04}") double minimumScore,
                         @Value("${rag.retrieval.top-k:10}") int retrievalTopK,
                         @Value("${rag.storage:memory}") String storage) {
        this.repository = repository; this.pipeline = pipeline; this.embeddings = embeddings;
        this.answerService = answerService; this.answerProviderReadiness = answerProviderReadiness;
        this.authReadiness = authReadiness;
        this.ingestion = ingestion; this.audit = audit; this.maxBytes = maxBytes; this.minimumScore = minimumScore; this.retrievalTopK = Math.max(3, Math.min(20, retrievalTopK)); this.storage = storage;
    }
    @GetMapping("/health")
    @Operation(summary = "Check application and storage readiness")
    public ResponseEntity<Map<String, String>> health() {
        Map<String, String> answerFields = answerProviderReadiness.asHealthFields();
        Map<String, String> authFields = authReadiness.asHealthFields();
        try {
            repository.isAvailable();
            var response = new HashMap<String, String>();
            response.put("status", "ok");
            response.put("storage", storage);
            response.put("embeddingProvider", embeddings.providerName());
            response.putAll(retrievalHealth());
            response.putAll(answerFields);
            response.putAll(authFields);
            return ResponseEntity.ok(response);
        } catch (Exception e) {
            var response = new HashMap<String, String>();
            response.put("status", "degraded");
            response.put("storage", storage);
            response.put("embeddingProvider", embeddings.providerName());
            response.putAll(retrievalHealth());
            response.putAll(answerFields);
            response.putAll(authFields);
            return ResponseEntity.status(503).body(response);
        }
    }
    @GetMapping("/csrf")
    @Operation(summary = "Get the CSRF token for browser form and API requests")
    public Map<String, String> csrf(CsrfToken token) { return Map.of("headerName", token.getHeaderName(), "parameterName", token.getParameterName(), "token", token.getToken()); }
    @GetMapping("/me")
    @PreAuthorize("isAuthenticated()")
    @Operation(summary = "Get the signed-in user and assigned roles")
    public Map<String, Object> me(Authentication authentication) {
        return Map.of("username", authentication.getName(), "roles", authentication.getAuthorities().stream().map(a -> a.getAuthority()).toList());
    }
    @GetMapping("/collections") public List<CollectionView> collections() { return repository.all().stream().map(this::view).toList(); }
    @PostMapping("/collections")
    @PreAuthorize("hasAnyRole('ADMIN','EDITOR')")
    @Operation(summary = "Create a collection")
    @SecurityRequirement(name = "sessionCookie")
    @SecurityRequirement(name = "csrfToken")
    public CollectionView create(@RequestBody CreateCollectionRequest request) {
        if (request == null || request.name() == null || request.name().isBlank()) throw new IllegalArgumentException("Collection name is required");
        String name = request.name().trim();
        if (name.length() > 80) throw new IllegalArgumentException("Collection name must be 80 characters or fewer");
        var collection = repository.create(name);
        audit.record("COLLECTION_CREATED", "collection", collection.id, "SUCCESS");
        return view(collection);
    }
    @DeleteMapping("/collections/{id}")
    @PreAuthorize("hasRole('ADMIN')")
    @Operation(summary = "Delete a collection and its documents")
    @SecurityRequirement(name = "sessionCookie")
    @SecurityRequirement(name = "csrfToken")
    public ResponseEntity<Void> deleteCollection(@PathVariable String id) {
        validateId(id);
        if (isHostedRetrieval()) {
            try { for (var document : repository.allDocuments(id)) hostedFileSearch.deleteFile(document.openaiFileId); }
            catch (Exception e) { throw new IllegalStateException("Could not remove the collection's hosted files", e); }
        }
        repository.deleteCollection(id); audit.record("COLLECTION_DELETED", "collection", id, "SUCCESS"); return ResponseEntity.noContent().build();
    }
    @GetMapping("/collections/{id}/documents")
    @Operation(summary = "List documents and ingestion states in a collection")
    public List<DocumentView> documents(@PathVariable String id) {
        validateId(id);
        repository.require(id);
        return repository.documents(id).stream().map(d -> new DocumentView(d.id(), d.name(), d.status(), d.chunkCount(), d.uploadedAt(), d.error())).toList();
    }
    @DeleteMapping("/collections/{id}/documents/{documentId}")
    @PreAuthorize("hasAnyRole('ADMIN','EDITOR')")
    @Operation(summary = "Delete a document and its indexed chunks")
    @SecurityRequirement(name = "sessionCookie")
    @SecurityRequirement(name = "csrfToken")
    public ResponseEntity<Void> deleteDocument(@PathVariable String id, @PathVariable String documentId) {
        validateId(id); validateId(documentId);
        if (isHostedRetrieval()) {
            try { hostedFileSearch.deleteFile(repository.document(id, documentId).openaiFileId); }
            catch (Exception e) { throw new IllegalStateException("Could not remove the hosted document", e); }
        }
        repository.deleteDocument(id, documentId); audit.record("DOCUMENT_DELETED", "document", documentId, "SUCCESS"); return ResponseEntity.noContent().build();
    }
    @PostMapping(value="/collections/{id}/documents", consumes=MediaType.MULTIPART_FORM_DATA_VALUE)
    @PreAuthorize("hasAnyRole('ADMIN','EDITOR')")
    @Operation(summary = "Upload and asynchronously index a PDF, DOCX, or TXT document")
    @SecurityRequirement(name = "sessionCookie")
    @SecurityRequirement(name = "csrfToken")
    public ResponseEntity<DocumentView> upload(@PathVariable String id, @RequestPart("file") MultipartFile file) throws Exception {
        validateId(id);
        repository.require(id);
        if (isHostedRetrieval() && (hostedFileSearch == null || !hostedFileSearch.available()))
            throw new IllegalStateException("Hosted File Search requires OPENAI_API_KEY and OPENAI_VECTOR_STORE_ID");
        if (file == null || file.isEmpty()) throw new IllegalArgumentException("Choose a non-empty document");
        if (file.getSize() > maxBytes) throw new IllegalArgumentException("File exceeds the configured upload limit");
        String name = Optional.ofNullable(file.getOriginalFilename()).orElse("document").replace('\\', '/');
        name = name.substring(name.lastIndexOf('/') + 1);
        String ext = name.contains(".") ? name.substring(name.lastIndexOf('.') + 1).toLowerCase(Locale.ROOT) : "";
        if (!ALLOWED.contains(ext)) throw new IllegalArgumentException("Supported formats are PDF, DOCX, and TXT");
        byte[] bytes = file.getBytes();
        String detectedType = new Tika().detect(bytes, name);
        boolean validType = switch (ext) {
            case "pdf" -> detectedType.equals("application/pdf");
            case "docx" -> detectedType.equals("application/vnd.openxmlformats-officedocument.wordprocessingml.document");
            case "txt" -> detectedType.startsWith("text/") || detectedType.equals("application/octet-stream");
            default -> false;
        };
        if (!validType) throw new IllegalArgumentException("The file contents do not match the ." + ext + " extension");
        String checksum = pipeline.checksum(bytes);
        DocumentData existing = repository.findByChecksum(id, checksum);
        if (existing != null && !repository.retryFailed(id, existing.id)) {
            audit.record("DOCUMENT_UPLOAD_DEDUPLICATED", "document", existing.id, existing.status);
            return ResponseEntity.ok(view(existing));
        }
        DocumentRegistration registration = existing == null ? repository.createPending(id, name, checksum) : new DocumentRegistration(existing, true);
        DocumentData pending = registration.document();
        if (!registration.created()) return ResponseEntity.ok(view(pending));
        try {
            Authentication user = SecurityContextHolder.getContext().getAuthentication();
            ingestion.process(id, pending.id, ext, bytes, user == null ? "unknown" : user.getName(), MDC.get("requestId"));
        } catch (Exception rejected) {
            repository.fail(id, pending.id, "Ingestion queue is busy. Retry in a moment.");
            throw new IllegalStateException("The ingestion queue is full; retry shortly");
        }
        audit.record("DOCUMENT_INGESTION_QUEUED", "document", pending.id, "PROCESSING");
        return ResponseEntity.accepted().body(view(pending));
    }
    @PostMapping("/collections/{id}/chat")
    @PreAuthorize("isAuthenticated()")
    @Operation(summary = "Ask a question grounded in one collection")
    @SecurityRequirement(name = "sessionCookie")
    @SecurityRequirement(name = "csrfToken")
    public ChatResponse chat(@PathVariable String id, @RequestBody ChatRequest request) {
        validateId(id);
        repository.require(id);
        if (request == null || request.question() == null || request.question().isBlank()) throw new IllegalArgumentException("Question is required");
        String question = request.question().trim();
        if (question.length() > 2000) throw new IllegalArgumentException("Question must be 2,000 characters or fewer");
        boolean broadPolicyList = java.util.regex.Pattern.compile("(?i)\\b(?:main|full|key|major|primary|all|list)\\b.{0,32}\\bpolic(?:y|ies)\\b|\\bpolic(?:y|ies)\\b.{0,32}\\b(?:main|full|key|major|primary|all|list)\\b").matcher(question).find();
        boolean broadSummary = broadPolicyList || java.util.regex.Pattern.compile("(?i)\\b(?:summari[sz]e|overview|highlights?)\\b|\\b(?:key|main)\\s+points?\\b").matcher(question).find();
        if (isHostedRetrieval()) {
            OpenAIFileSearchService.SearchAnswer hosted;
            try { hosted = hostedFileSearch.answer(question, id, broadSummary ? 20 : retrievalTopK); }
            catch (Exception e) { throw new IllegalStateException("Hosted document search is temporarily unavailable", e); }
            Map<String, DocumentData> documentsByFile = new HashMap<>();
            for (var document : repository.allDocuments(id)) if (document.openaiFileId != null) documentsByFile.put(document.openaiFileId, document);
            var citations = new ArrayList<Citation>();
            for (var result : hosted.results().stream().limit(retrievalTopK).toList()) {
                var document = documentsByFile.get(Objects.toString(result.get("file_id"), ""));
                if (document == null || !"READY".equals(document.status)) continue;
                String excerpt = "";
                if (result.get("content") instanceof List<?> content) for (Object part : content)
                    if (part instanceof Map<?, ?> map && "text".equals(map.get("type"))) excerpt += Objects.toString(map.get("text"), "") + " ";
                excerpt = excerpt.isBlank() ? document.name : excerpt.strip();
                citations.add(new Citation(document.id, document.name, citations.size() + 1, null, excerpt.length() <= 220 ? excerpt : excerpt.substring(0, 217) + "…"));
            }
            if (citations.isEmpty() || hosted.answer().isBlank()) {
                boolean processing = repository.documents(id).stream().anyMatch(d -> "PROCESSING".equals(d.status()));
                String message = processing ? "Documents are still being processed. Try again in a moment." : "I couldn't find enough information in this collection to answer that.";
                audit.record("QUESTION_ANSWERED", "collection", id, "NO_EVIDENCE");
                return new ChatResponse(citations.isEmpty() || hosted.answer().isBlank() ? message : hosted.answer(), citations, false);
            }
            audit.record("QUESTION_ANSWERED", "collection", id, "GROUNDED");
            return new ChatResponse(hosted.answer(), citations, true);
        }
        var matches = broadSummary
            ? repository.summaryCandidates(id, Integer.MAX_VALUE)
            : repository.search(id, embeddings.embed(question), retrievalTopK, minimumScore);
        if (matches.isEmpty()) {
            boolean processing = repository.documents(id).stream().anyMatch(d -> "PROCESSING".equals(d.status()));
            String message = processing ? "Documents are still being processed. Try again in a moment." : "I couldn't find enough information in this collection to answer that.";
            audit.record("QUESTION_ANSWERED", "collection", id, processing ? "PROCESSING" : "NO_EVIDENCE");
            return new ChatResponse(message, List.of(), false);
        }
        var context = matches.stream().map(m -> "[" + m.document().name + ", chunk " + m.chunk().index() + (m.chunk().pageNumber() == null ? "" : ", page " + m.chunk().pageNumber()) + "]\n" + m.chunk().text()).toList();
        String answer = answerService.answer(question, context);
        // Summaries process every chunk, but keep the response's citation list compact.
        var citations = matches.stream().limit(broadSummary ? retrievalTopK : Integer.MAX_VALUE)
            .map(m -> new Citation(m.document().id, m.document().name, m.chunk().index(), m.chunk().pageNumber(), excerpt(m.chunk().text()))).toList();
        audit.record("QUESTION_ANSWERED", "collection", id, "GROUNDED");
        return new ChatResponse(answer, citations, true);
    }
    private String excerpt(String text) { return text.length() <= 220 ? text : text.substring(0, 217) + "…"; }
    private void validateId(String value) {
        try { UUID.fromString(value); }
        catch (IllegalArgumentException e) { throw new IllegalArgumentException("Identifier must be a UUID"); }
    }
    private CollectionView view(CollectionData c) { return new CollectionView(c.id, c.name, c.createdAt, c.documentCount); }
    private DocumentView view(DocumentData d) { return new DocumentView(d.id, d.name, d.status, d.chunkCount, d.uploadedAt, d.error); }
    private boolean isHostedRetrieval() { return "openai-file-search".equalsIgnoreCase(retrievalProvider); }
    private Map<String, String> retrievalHealth() {
        if (!Set.of("local", "openai-file-search").contains(Objects.toString(retrievalProvider, "local").toLowerCase(Locale.ROOT)))
            return Map.of("retrievalProvider", "invalid", "retrievalProviderStatus", "degraded", "retrievalProviderMessage", "RAG_RETRIEVAL_PROVIDER must be local or openai-file-search.");
        if (isHostedRetrieval() && (hostedFileSearch == null || !hostedFileSearch.available()))
            return Map.of("retrievalProvider", retrievalProvider, "retrievalProviderStatus", "degraded", "retrievalProviderMessage", "OPENAI_API_KEY and OPENAI_VECTOR_STORE_ID are required for hosted File Search.");
        return Map.of("retrievalProvider", Objects.toString(retrievalProvider, "local"), "retrievalProviderStatus", "ok", "retrievalProviderMessage", isHostedRetrieval() ? "Hosted File Search is configured; connectivity is checked when used." : "Local retrieval is enabled.");
    }
}
