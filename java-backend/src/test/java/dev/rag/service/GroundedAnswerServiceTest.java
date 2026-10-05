package dev.rag.service;

import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.ObjectProvider;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

class GroundedAnswerServiceTest {
    @Test
    void usesExtractiveAnswerWhenNoProviderKeyIsConfigured() {
        var service = service("", "auto", null);

        String answer = service.answer("What is the retention policy?", List.of(
                "[policy.txt, chunk 1] The retention policy requires teams to review records every year."));

        assertTrue(answer.contains("review records every year"));
    }

    @Test
    void returnsInsufficientInformationWhenPassagesDoNotSupportQuestion() {
        var service = service("", "auto", null);

        String answer = service.answer("What is the retention policy?", List.of("[other.txt] The office is near the river."));

        assertTrue(answer.contains("I don't have enough information"));
    }

    @Test
    void returnsInsufficientInformationWhenQuestionHasNoSearchableTerms() {
        var service = service("", "auto", null);

        String answer = service.answer("the and of", List.of(
                "[policy.txt] The retention policy requires teams to review records every year."));

        assertTrue(answer.contains("I don't have enough information"));
    }

    @Test
    void fullPolicyListReturnsSeveralRelevantPolicyAreas() {
        var service = service("", "auto", null);
        String answer = service.answer("What is the full list of main policies?", List.of(
                "[handbook, chunk 1] Employees receive overtime pay for approved extra hours.",
                "[handbook, chunk 2] Annual leave requests need manager approval before time off.",
                "[handbook, chunk 3] Employees must report workplace safety hazards to their manager promptly.",
                "[handbook, chunk 4] Employees may report harassment directly to Human Resources.",
                "[handbook, chunk 5] Staff must record attendance and notify managers about absences."));

        assertTrue(answer.contains("overtime pay"));
        assertTrue(answer.contains("Annual leave"));
        assertTrue(answer.contains("safety hazards"));
        assertTrue(answer.contains("report harassment"));
        assertTrue(answer.contains("record attendance"));
    }

    @Test
    void genericSummaryReturnsSeveralRetrievedPointsWithoutKeywordOverlap() {
        var service = service("", "auto", null);
        String answer = service.answer("Summarize the key points", List.of(
                "[handbook, chunk 1] Staff can request annual leave through their manager.",
                "[handbook, chunk 2] Employees must record working hours accurately.",
                "[handbook, chunk 3] Report safety incidents to a supervisor as soon as possible."));

        assertTrue(answer.contains("annual leave"));
        assertTrue(answer.contains("working hours"));
        assertTrue(answer.contains("safety incidents"));
        assertFalse(answer.contains("I don't have enough information"));
    }

    @Test
    void genericSummarySkipsLowercaseChunkFragmentsAndUsesSummaryHeading() {
        var service = service("", "auto", null);
        String answer = service.answer("Summarize the key points", List.of(
                "[handbook, chunk 1] Annual leave: Employees receive twenty paid days each year.",
                "[handbook, chunk 2] roval from Human Resources. Employees should retain approval notices.",
                "[handbook, chunk 3] Employees must report safety incidents to a supervisor promptly."));

        assertTrue(answer.startsWith("Key takeaways:"));
        assertTrue(answer.contains("Annual leave:"));
        assertTrue(answer.contains("report safety incidents"));
        assertFalse(answer.contains("roval from Human Resources"));
    }

    @Test
    void focusedTwoWordQuestionDoesNotReturnPassagesMatchingOnlyOneTerm() {
        var service = service("", "auto", null);
        String answer = service.answer("Annual Leave", List.of(
                "[handbook, chunk 1] Parental leave may be arranged through Human Resources.",
                "[handbook, chunk 2] Annual leave requests should be submitted to a manager."));

        assertTrue(answer.contains("requests should be submitted to a manager"));
        assertFalse(answer.contains("Parental leave"));
    }

    @Test
    void usesSpringAiAgentWhenConfiguredAndAvailable() {
        var agent = mock(SpringAiGroundedAgentService.class);
        var service = service("secret", "spring-ai", agent);
        var passages = List.of("Source passage");
        when(agent.answer("question", passages)).thenReturn("Grounded response");

        assertEquals("Grounded response", service.answer("question", passages));
        verify(agent).answer("question", passages);
    }

    @Test
    void sendsBroadSummaryToBatchSummarizerWhenProviderIsConfigured() {
        var agent = mock(SpringAiGroundedAgentService.class);
        var service = service("secret", "auto", agent);
        var passages = List.of("chunk one", "chunk two");
        when(agent.summarize("Summarize the key points", passages)).thenReturn("Consolidated summary");

        assertEquals("Consolidated summary", service.answer("Summarize the key points", passages));
        verify(agent).summarize("Summarize the key points", passages);
    }

    @Test
    void fallsBackToExtractiveAnswerWhenRequiredSpringAiProviderFails() {
        var agent = mock(SpringAiGroundedAgentService.class);
        var service = service("secret", "spring-ai", agent);
        var passages = List.of("[policy.txt] The retention policy requires teams to review records every year.");
        when(agent.answer("What is the retention policy?", passages)).thenThrow(new IllegalStateException("provider down"));

        String answer = service.answer("What is the retention policy?", passages);

        assertTrue(answer.contains("review records every year"));
    }

    @SuppressWarnings("unchecked")
    private GroundedAnswerService service(String apiKey, String provider, SpringAiGroundedAgentService agent) {
        ObjectProvider<SpringAiGroundedAgentService> agentProvider = mock(ObjectProvider.class);
        when(agentProvider.getIfAvailable()).thenReturn(agent);
        return new GroundedAnswerService(apiKey, "model", "http://localhost", provider, agentProvider);
    }
}
