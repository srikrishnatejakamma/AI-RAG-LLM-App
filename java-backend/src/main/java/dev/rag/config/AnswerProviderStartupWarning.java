package dev.rag.config;

import dev.rag.service.AnswerProviderReadinessService;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.context.event.EventListener;
import org.springframework.stereotype.Component;
import org.springframework.boot.context.event.ApplicationReadyEvent;

@Component
public class AnswerProviderStartupWarning {
    private static final Logger log = LoggerFactory.getLogger(AnswerProviderStartupWarning.class);
    private final AnswerProviderReadinessService readinessService;

    public AnswerProviderStartupWarning(AnswerProviderReadinessService readinessService) {
        this.readinessService = readinessService;
    }

    @EventListener(ApplicationReadyEvent.class)
    public void logReadiness() {
        var status = readinessService.status();
        if ("degraded".equals(status.status())) {
            log.warn("Answer provider startup check: {} (provider={})", status.message(), status.provider());
        } else {
            log.info("Answer provider startup check: {} (provider={})", status.message(), status.provider());
        }
    }
}
