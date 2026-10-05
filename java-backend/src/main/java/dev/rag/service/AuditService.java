package dev.rag.service;

import dev.rag.store.RagRepository;
import dev.rag.store.RagRepository.AuditRecord;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.slf4j.MDC;
import org.springframework.security.core.Authentication;
import org.springframework.security.core.context.SecurityContextHolder;
import org.springframework.stereotype.Service;
import java.time.Instant;

@Service
public class AuditService {
    private static final Logger log = LoggerFactory.getLogger(AuditService.class);
    private final RagRepository repository;
    public AuditService(RagRepository repository) { this.repository = repository; }
    public AuditRecord record(String action, String resourceType, String resourceId, String outcome) {
        Authentication authentication = SecurityContextHolder.getContext().getAuthentication();
        String actor = authentication != null && authentication.isAuthenticated() ? authentication.getName() : "anonymous";
        return record(actor, action, resourceType, resourceId, outcome, MDC.get("requestId"));
    }
    public AuditRecord record(String actor, String action, String resourceType, String resourceId, String outcome, String requestId) {
        var entry = new AuditRecord(0, Instant.now(), safe(actor), safe(action), safe(resourceType), safe(resourceId), safe(outcome), safe(requestId));
        try {
            var saved = repository.recordAudit(entry);
            log.atInfo().addKeyValue("audit", true).addKeyValue("eventId", saved.eventId())
                .addKeyValue("actor", saved.actor()).addKeyValue("action", saved.action())
                .addKeyValue("resourceType", saved.resourceType()).addKeyValue("resourceId", saved.resourceId())
                .addKeyValue("outcome", saved.outcome()).addKeyValue("requestId", saved.requestId())
                .log("Audit event");
            return saved;
        } catch (Exception e) {
            log.error("Audit persistence failed for action {} on resource {}", action, resourceId, e);
            return entry;
        }
    }
    private String safe(String value) {
        if (value == null) return "";
        return value.length() <= 120 ? value : value.substring(0, 120);
    }
}
