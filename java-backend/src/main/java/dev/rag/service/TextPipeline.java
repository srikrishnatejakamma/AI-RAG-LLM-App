package dev.rag.service;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;
import java.security.MessageDigest;
import java.util.*;
import java.util.regex.Pattern;

@Service
public class TextPipeline {
    private static final Pattern WORD = Pattern.compile("[\\p{L}\\p{N}]{2,}");
    private final int chunkSize;
    private final int overlap;
    public TextPipeline(@Value("${rag.chunk-size:1200}") int chunkSize, @Value("${rag.chunk-overlap:180}") int overlap) {
        if (chunkSize < 200 || overlap < 0 || overlap >= chunkSize) throw new IllegalArgumentException("Chunk size must be at least 200 and overlap must be smaller than chunk size");
        this.chunkSize = chunkSize; this.overlap = overlap;
    }
    public String checksum(byte[] data) throws Exception {
        return HexFormat.of().formatHex(MessageDigest.getInstance("SHA-256").digest(data));
    }
    public List<String> chunkTexts(String text) {
        var result = new ArrayList<String>();
        var normalized = text.replaceAll("\\s+", " ").trim();
        if (normalized.isEmpty()) return result;
        int start = 0;
        while (start < normalized.length()) {
            int end = Math.min(normalized.length(), start + chunkSize);
            if (end < normalized.length()) {
                int boundary = normalized.lastIndexOf(' ', end);
                if (boundary > start + chunkSize / 2) end = boundary;
            }
            String part = normalized.substring(start, end).trim();
            if (!part.isEmpty()) result.add(part);
            if (end >= normalized.length()) break;
            start = Math.max(start + 1, end - overlap);
        }
        return result;
    }
    // Local deterministic feature hashing is a zero-credential fallback.
    public float[] embedLocal(String text) {
        float[] vector = new float[1536];
        var matcher = WORD.matcher(text.toLowerCase(Locale.ROOT));
        while (matcher.find()) {
            String token = matcher.group();
            int hash = token.hashCode();
            int index = Math.floorMod(hash, vector.length);
            vector[index] += (hash & 0x100) == 0 ? 1f : -1f;
        }
        float norm = 0;
        for (float value : vector) norm += value * value;
        if (norm > 0) {
            float scale = (float)(1 / Math.sqrt(norm));
            for (int i = 0; i < vector.length; i++) vector[i] *= scale;
        }
        return vector;
    }
}
