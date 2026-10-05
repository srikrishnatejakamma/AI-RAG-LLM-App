package dev.rag.service;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

class AnswerProviderReadinessServiceTest {
    @Test
    void normalizesProviderAndReportsMissingKeyAsDegraded() {
        var service = new AnswerProviderReadinessService(" AUTO ", " ");

        var status = service.status();

        assertEquals("auto", status.provider());
        assertEquals("degraded", status.status());
        assertEquals("OPENAI_API_KEY is missing; chat will run in extractive fallback mode.", status.message());
    }

    @Test
    void rejectsUnknownProviderAndReportsConfiguredProvider() {
        var invalid = new AnswerProviderReadinessService("custom", "key").status();
        var configured = new AnswerProviderReadinessService("legacy", "key").status();

        assertEquals("degraded", invalid.status());
        assertEquals("legacy", configured.provider());
        assertEquals("ok", configured.status());
    }
}
