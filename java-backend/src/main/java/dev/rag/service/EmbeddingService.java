package dev.rag.service;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;
import org.springframework.http.client.SimpleClientHttpRequestFactory;
import org.springframework.web.client.RestClient;
import java.util.*;

@Service
public class EmbeddingService {
    private final TextPipeline local;
    private final String apiKey;
    private final String model;
    private final String provider;
    private final String baseUrl;
    private final RestClient http;
    public EmbeddingService(TextPipeline local,
                            @Value("${rag.openai.api-key:}") String apiKey,
                            @Value("${rag.openai.embedding-model:text-embedding-3-small}") String model,
                            @Value("${rag.embedding.provider:local}") String provider,
                            @Value("${rag.openai.base-url:https://api.openai.com/v1}") String baseUrl) {
        this.local = local; this.apiKey = apiKey; this.model = model; this.provider = provider; this.baseUrl = baseUrl;
        if (!"local".equalsIgnoreCase(provider) && !"openai".equalsIgnoreCase(provider)) throw new IllegalArgumentException("RAG_EMBEDDING_PROVIDER must be local or openai");
        this.http = createClient(baseUrl);
    }
    @SuppressWarnings("unchecked")
    public List<float[]> embedAll(List<String> texts) {
        if (texts.isEmpty()) return List.of();
        if (!"openai".equalsIgnoreCase(provider)) return texts.stream().map(local::embedLocal).toList();
        if (apiKey == null || apiKey.isBlank()) throw new IllegalStateException("OPENAI_API_KEY is required when RAG_EMBEDDING_PROVIDER=openai");
        var result = new ArrayList<float[]>(texts.size());
        for (int start = 0; start < texts.size(); start += 64) {
            result.addAll(embedBatch(texts.subList(start, Math.min(start + 64, texts.size()))));
        }
        return List.copyOf(result);
    }
    @SuppressWarnings("unchecked")
    private List<float[]> embedBatch(List<String> texts) {
        try {
            var payload = new HashMap<String,Object>();
            payload.put("model", model); payload.put("input", texts);
            if (model.startsWith("text-embedding-3")) payload.put("dimensions", 1536);
            Map<String,Object> response = http.post().uri("embeddings")
                .header("Authorization", "Bearer " + apiKey)
                .body(payload).retrieve().body(Map.class);
            var items = (List<Map<String,Object>>) response.get("data");
            if (items == null || items.size() != texts.size()) throw new IllegalStateException("Embedding provider returned an incomplete batch");
            var ordered = new ArrayList<float[]>(Collections.nCopies(texts.size(), null));
            for (var item : items) {
                int index = ((Number)item.get("index")).intValue();
                var values = (List<Number>)item.get("embedding");
                float[] vector = new float[values.size()];
                if (vector.length != 1536) throw new IllegalStateException("Embedding model must return 1,536 dimensions for this vector index");
                for (int i = 0; i < values.size(); i++) vector[i] = values.get(i).floatValue();
                if (index < 0 || index >= ordered.size()) throw new IllegalStateException("Embedding provider returned invalid item indexes");
                ordered.set(index, vector);
            }
            if (ordered.stream().anyMatch(Objects::isNull)) throw new IllegalStateException("Embedding provider returned invalid item indexes");
            return List.copyOf(ordered);
        } catch (Exception e) {
            throw new IllegalStateException("Embedding provider is unavailable", e);
        }
    }
    public float[] embed(String text) { return embedAll(List.of(text)).getFirst(); }
    public String providerName() { return "openai".equalsIgnoreCase(provider) ? "openai" : "local"; }
    private static RestClient createClient(String baseUrl) {
        var factory = new SimpleClientHttpRequestFactory();
        factory.setConnectTimeout(5000); factory.setReadTimeout(45000);
        return RestClient.builder().baseUrl(baseUrl).requestFactory(factory).build();
    }
}
