package dev.rag.service;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;

import java.util.Locale;
import java.util.Map;

@Service
public class AnswerProviderReadinessService {
    private final String answerProvider;
    private final String apiKey;

    public AnswerProviderReadinessService(@Value("${rag.answer.provider:auto}") String answerProvider,
                                          @Value("${rag.openai.api-key:}") String apiKey) {
        this.answerProvider = answerProvider;
        this.apiKey = apiKey;
    }

    public AnswerProviderStatus status() {
        String provider = normalizedProvider();
        if (!provider.equals("auto") && !provider.equals("spring-ai") && !provider.equals("legacy")) {
            return new AnswerProviderStatus(provider, "degraded", "RAG_ANSWER_PROVIDER must be auto, spring-ai, or legacy.");
        }
        if (apiKey == null || apiKey.isBlank()) {
            return new AnswerProviderStatus(provider, "degraded", "OPENAI_API_KEY is missing; chat will run in extractive fallback mode.");
        }
        return new AnswerProviderStatus(provider, "ok", "OPENAI_API_KEY is configured. Provider connectivity is verified at runtime.");
    }

    public Map<String, String> asHealthFields() {
        AnswerProviderStatus status = status();
        return Map.of(
            "answerProvider", status.provider(),
            "answerProviderStatus", status.status(),
            "answerProviderMessage", status.message()
        );
    }

    private String normalizedProvider() {
        return answerProvider == null ? "auto" : answerProvider.trim().toLowerCase(Locale.ROOT);
    }

    public record AnswerProviderStatus(String provider, String status, String message) {}
}
