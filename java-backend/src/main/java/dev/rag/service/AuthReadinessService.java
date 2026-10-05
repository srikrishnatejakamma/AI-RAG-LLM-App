package dev.rag.service;

import org.springframework.stereotype.Service;

import java.util.Map;

@Service
public class AuthReadinessService {
    private volatile boolean oneTimeAdminCredentials;
    private volatile String adminUsername = "admin";

    public void setOneTimeAdminCredentials(boolean oneTimeAdminCredentials, String adminUsername) {
        this.oneTimeAdminCredentials = oneTimeAdminCredentials;
        if (adminUsername != null && !adminUsername.isBlank()) this.adminUsername = adminUsername;
    }

    public Map<String, String> asHealthFields() {
        return Map.of(
            "authOneTimeAdminCredentials", Boolean.toString(oneTimeAdminCredentials),
            "authAdminUsername", adminUsername
        );
    }
}
