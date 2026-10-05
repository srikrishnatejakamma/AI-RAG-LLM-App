package dev.rag.config;

import dev.rag.service.AuditService;
import dev.rag.service.AuthReadinessService;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.http.HttpMethod;
import org.springframework.http.HttpStatus;
import org.springframework.security.config.annotation.method.configuration.EnableMethodSecurity;
import org.springframework.security.config.annotation.web.builders.HttpSecurity;
import org.springframework.security.core.userdetails.User;
import org.springframework.security.core.userdetails.UserDetails;
import org.springframework.security.core.userdetails.UserDetailsService;
import org.springframework.security.crypto.bcrypt.BCryptPasswordEncoder;
import org.springframework.security.crypto.password.PasswordEncoder;
import org.springframework.security.provisioning.InMemoryUserDetailsManager;
import org.springframework.security.web.SecurityFilterChain;
import org.springframework.security.web.csrf.CookieCsrfTokenRepository;
import org.springframework.security.web.csrf.CsrfTokenRequestAttributeHandler;
import org.springframework.security.web.util.matcher.AntPathRequestMatcher;
import java.security.SecureRandom;
import java.util.*;

@Configuration
@EnableMethodSecurity
public class SecurityConfiguration {
    private static final Logger log = LoggerFactory.getLogger(SecurityConfiguration.class);

    @Bean PasswordEncoder passwordEncoder() { return new BCryptPasswordEncoder(12); }

    @Bean UserDetailsService userDetailsService(PasswordEncoder encoder,
        AuthReadinessService authReadinessService,
        @Value("${RAG_ADMIN_USERNAME:admin}") String adminName,
        @Value("${RAG_ADMIN_PASSWORD:}") String adminPassword,
        @Value("${RAG_EDITOR_USERNAME:}") String editorName,
        @Value("${RAG_EDITOR_PASSWORD:}") String editorPassword,
        @Value("${RAG_VIEWER_USERNAME:}") String viewerName,
        @Value("${RAG_VIEWER_PASSWORD:}") String viewerPassword,
        @Value("${rag.storage:memory}") String storage) {
        String bootstrapPassword = adminPassword;
        if (bootstrapPassword.isBlank()) {
            if ("postgres".equalsIgnoreCase(storage)) throw new IllegalStateException("Set RAG_ADMIN_PASSWORD before starting persistent mode");
            byte[] bytes = new byte[24]; new SecureRandom().nextBytes(bytes);
            bootstrapPassword = Base64.getUrlEncoder().withoutPadding().encodeToString(bytes);
            log.warn("Generated one-time local admin password. Username: {} Password: {}. Set RAG_ADMIN_PASSWORD to keep credentials across restarts.", adminName, bootstrapPassword);
            authReadinessService.setOneTimeAdminCredentials(true, adminName);
        } else {
            authReadinessService.setOneTimeAdminCredentials(false, adminName);
        }
        var users = new ArrayList<UserDetails>();
        users.add(user(adminName, bootstrapPassword, "ADMIN", encoder));
        addOptionalUser(users, editorName, editorPassword, "EDITOR", encoder);
        addOptionalUser(users, viewerName, viewerPassword, "VIEWER", encoder);
        return new InMemoryUserDetailsManager(users);
    }

    @Bean SecurityFilterChain applicationSecurity(HttpSecurity http, AuditService audit) throws Exception {
        var csrfRepository = CookieCsrfTokenRepository.withHttpOnlyFalse();
        csrfRepository.setCookiePath("/");
        var csrfHandler = new CsrfTokenRequestAttributeHandler();
        csrfHandler.setCsrfRequestAttributeName("_csrf");
        http
            .csrf(csrf -> csrf.csrfTokenRepository(csrfRepository).csrfTokenRequestHandler(csrfHandler))
            .authorizeHttpRequests(authorize -> authorize
                .requestMatchers(HttpMethod.OPTIONS, "/**").permitAll()
                .requestMatchers("/", "/index.html", "/app.css", "/app.js", "/favicon.ico", "/error").permitAll()
                .requestMatchers("/react", "/react/**").permitAll()
                .requestMatchers("/api/health", "/api/csrf").permitAll()
                .requestMatchers("/v3/api-docs/**", "/swagger-ui/**", "/swagger-ui.html").permitAll()
                .requestMatchers("/api/admin/**").hasRole("ADMIN")
                .requestMatchers(HttpMethod.POST, "/api/collections").hasAnyRole("ADMIN", "EDITOR")
                .requestMatchers(HttpMethod.DELETE, "/api/collections/*").hasRole("ADMIN")
                .requestMatchers(HttpMethod.POST, "/api/collections/*/documents").hasAnyRole("ADMIN", "EDITOR")
                .requestMatchers(HttpMethod.DELETE, "/api/collections/*/documents/*").hasAnyRole("ADMIN", "EDITOR")
                .requestMatchers("/api/**").authenticated()
                .anyRequest().authenticated())
            .exceptionHandling(exceptions -> exceptions
                .defaultAuthenticationEntryPointFor((request, response, failure) -> {
                    response.setStatus(HttpStatus.UNAUTHORIZED.value()); response.setContentType("application/json");
                    response.getWriter().write("{\"error\":\"Sign in is required\"}");
                }, new AntPathRequestMatcher("/api/**"))
                .accessDeniedHandler((request, response, denied) -> {
                    response.setStatus(HttpStatus.FORBIDDEN.value()); response.setContentType("application/json");
                    response.getWriter().write("{\"error\":\"Your role cannot perform this action\"}");
                }))
            .formLogin(form -> form.loginProcessingUrl("/login")
                .successHandler((request, response, authentication) -> {
                    audit.record(authentication.getName(), "AUTHENTICATION_SUCCEEDED", "session", null, "SUCCESS", org.slf4j.MDC.get("requestId"));
                    response.setStatus(HttpStatus.NO_CONTENT.value());
                })
                .failureHandler((request, response, failure) -> {
                    String attempted = request.getParameter("username");
                    audit.record(attempted == null ? "unknown" : attempted, "AUTHENTICATION_FAILED", "session", null, "DENIED", org.slf4j.MDC.get("requestId"));
                    response.sendError(HttpStatus.UNAUTHORIZED.value());
                }).permitAll())
            .logout(logout -> logout.logoutUrl("/logout")
                .logoutSuccessHandler((request, response, authentication) -> {
                    if (authentication != null) audit.record(authentication.getName(), "LOGOUT", "session", null, "SUCCESS", org.slf4j.MDC.get("requestId"));
                    response.setStatus(HttpStatus.NO_CONTENT.value());
                }).deleteCookies("JSESSIONID", "XSRF-TOKEN"));
        return http.build();
    }

    private static UserDetails user(String username, String password, String role, PasswordEncoder encoder) {
        return User.withUsername(username).password(encoder.encode(password)).roles(role).build();
    }
    private static void addOptionalUser(List<UserDetails> users, String username, String password, String role, PasswordEncoder encoder) {
        if (username.isBlank() && password.isBlank()) return;
        if (username.isBlank() || password.isBlank()) throw new IllegalStateException("Both username and password must be set for the " + role + " account");
        users.add(user(username, password, role, encoder));
    }
}
