package dev.rag.api;

import org.junit.jupiter.api.Test;
import org.springframework.web.multipart.MaxUploadSizeExceededException;
import org.springframework.web.multipart.support.MissingServletRequestPartException;

import java.util.NoSuchElementException;

import static org.junit.jupiter.api.Assertions.assertEquals;

class ApiExceptionHandlerTest {
    private final ApiExceptionHandler handler = new ApiExceptionHandler();

    @Test
    void mapsMissingResourcesToNotFound() {
        var response = handler.missing(new NoSuchElementException("Collection not found"));

        assertEquals(404, response.getStatusCode().value());
        assertEquals("Collection not found", response.getBody().error());
    }

    @Test
    void mapsInvalidAndMalformedRequestsToBadRequest() {
        var invalid = handler.invalid(new IllegalArgumentException("Question is required"));
        var malformed = handler.malformed(new MissingServletRequestPartException("file"));

        assertEquals(400, invalid.getStatusCode().value());
        assertEquals("Question is required", invalid.getBody().error());
        assertEquals(400, malformed.getStatusCode().value());
        assertEquals("Request body or required upload field is invalid", malformed.getBody().error());
    }

    @Test
    void mapsQueueFailuresAndOversizedUploads() {
        var unavailable = handler.unavailable(new IllegalStateException("The ingestion queue is full; retry shortly"));
        var tooLarge = handler.tooLarge(new MaxUploadSizeExceededException(1024));

        assertEquals(503, unavailable.getStatusCode().value());
        assertEquals("The ingestion queue is full; retry shortly", unavailable.getBody().error());
        assertEquals(413, tooLarge.getStatusCode().value());
        assertEquals("Upload exceeds the configured file size limit", tooLarge.getBody().error());
    }

    @Test
    void hidesUnexpectedExceptionDetails() {
        var response = handler.failure(new RuntimeException("database password should not leak"));

        assertEquals(500, response.getStatusCode().value());
        assertEquals("The request could not be processed", response.getBody().error());
    }
}
