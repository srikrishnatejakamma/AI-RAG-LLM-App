package dev.rag.config;

import io.swagger.v3.oas.annotations.OpenAPIDefinition;
import io.swagger.v3.oas.annotations.enums.SecuritySchemeIn;
import io.swagger.v3.oas.annotations.enums.SecuritySchemeType;
import io.swagger.v3.oas.annotations.info.Info;
import io.swagger.v3.oas.annotations.security.SecurityScheme;
import org.springframework.context.annotation.Configuration;

@Configuration
@OpenAPIDefinition(info = @Info(title = "Gather RAG API", version = "1.0.0", description = "Collection-scoped document ingestion, retrieval, and grounded question answering."))
@SecurityScheme(name = "sessionCookie", type = SecuritySchemeType.APIKEY, in = SecuritySchemeIn.COOKIE, paramName = "JSESSIONID", description = "Sign in through the app to establish a session.")
@SecurityScheme(name = "csrfToken", type = SecuritySchemeType.APIKEY, in = SecuritySchemeIn.HEADER, paramName = "X-XSRF-TOKEN", description = "Read the token from the XSRF-TOKEN cookie for state-changing requests.")
public class OpenApiConfiguration {}
