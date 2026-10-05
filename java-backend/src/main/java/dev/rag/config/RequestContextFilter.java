package dev.rag.config;

import jakarta.servlet.FilterChain;
import jakarta.servlet.ServletException;
import jakarta.servlet.http.HttpServletRequest;
import jakarta.servlet.http.HttpServletResponse;
import org.slf4j.MDC;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.core.Ordered;
import org.springframework.core.annotation.Order;
import org.springframework.stereotype.Component;
import org.springframework.web.filter.OncePerRequestFilter;
import java.io.IOException;
import java.util.UUID;

@Component("requestAuditContextFilter")
@Order(Ordered.HIGHEST_PRECEDENCE)
public class RequestContextFilter extends OncePerRequestFilter {
    private static final Logger log = LoggerFactory.getLogger(RequestContextFilter.class);
    @Override protected void doFilterInternal(HttpServletRequest request, HttpServletResponse response, FilterChain chain)
        throws ServletException, IOException {
        String requestId = UUID.randomUUID().toString();
        long started = System.nanoTime();
        MDC.put("requestId", requestId);
        response.setHeader("X-Request-ID", requestId);
        try { chain.doFilter(request, response); }
        finally {
            if (request.getRequestURI().startsWith("/api/") || request.getRequestURI().equals("/login") || request.getRequestURI().equals("/logout")) {
                log.atInfo().addKeyValue("event", "http_request").addKeyValue("method", request.getMethod())
                    .addKeyValue("path", request.getRequestURI()).addKeyValue("status", response.getStatus())
                    .addKeyValue("durationMs", (System.nanoTime() - started) / 1_000_000).addKeyValue("requestId", requestId)
                    .log("HTTP request");
            }
            MDC.remove("requestId");
        }
    }
}
