package dev.rag.api;

import dev.rag.store.RagRepository;
import dev.rag.store.RagRepository.AuditRecord;
import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.security.SecurityRequirement;
import io.swagger.v3.oas.annotations.tags.Tag;
import org.springframework.security.access.prepost.PreAuthorize;
import org.springframework.web.bind.annotation.*;
import java.util.List;

@RestController
@RequestMapping("/api/admin/audit")
@Tag(name = "Administration", description = "Administrative audit records")
public class AdminAuditController {
    public record AuditPage(List<AuditRecord> items, long nextCursor) {}
    private final RagRepository repository;
    public AdminAuditController(RagRepository repository) { this.repository = repository; }
    @GetMapping
    @PreAuthorize("hasRole('ADMIN')")
    @Operation(summary = "List audit events", description = "Returns newest events first. Use nextCursor as beforeEventId to fetch the next page.")
    @SecurityRequirement(name = "sessionCookie")
    public AuditPage events(@RequestParam(defaultValue = "100") int limit, @RequestParam(defaultValue = "9223372036854775807") long beforeEventId) {
        int safeLimit = Math.max(1, Math.min(limit, 500));
        var events = repository.auditEvents(safeLimit, beforeEventId);
        long next = events.size() == safeLimit ? events.getLast().eventId() : 0;
        return new AuditPage(events, next);
    }
}
