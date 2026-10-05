package dev.rag.api;

import dev.rag.model.Models.ErrorResponse;
import org.springframework.http.*;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MaxUploadSizeExceededException;
import org.springframework.http.converter.HttpMessageNotReadableException;
import org.springframework.web.multipart.support.MissingServletRequestPartException;
import java.util.NoSuchElementException;

@RestControllerAdvice
public class ApiExceptionHandler {
    @ExceptionHandler(NoSuchElementException.class)
    ResponseEntity<ErrorResponse> missing(NoSuchElementException e) { return ResponseEntity.status(404).body(new ErrorResponse(e.getMessage())); }
    @ExceptionHandler(IllegalArgumentException.class)
    ResponseEntity<ErrorResponse> invalid(IllegalArgumentException e) { return ResponseEntity.badRequest().body(new ErrorResponse(e.getMessage())); }
    @ExceptionHandler({HttpMessageNotReadableException.class, MissingServletRequestPartException.class})
    ResponseEntity<ErrorResponse> malformed(Exception e) { return ResponseEntity.badRequest().body(new ErrorResponse("Request body or required upload field is invalid")); }
    @ExceptionHandler(IllegalStateException.class)
    ResponseEntity<ErrorResponse> unavailable(IllegalStateException e) { return ResponseEntity.status(503).body(new ErrorResponse(e.getMessage())); }
    @ExceptionHandler(MaxUploadSizeExceededException.class)
    ResponseEntity<ErrorResponse> tooLarge(MaxUploadSizeExceededException e) { return ResponseEntity.status(413).body(new ErrorResponse("Upload exceeds the configured file size limit")); }
    @ExceptionHandler(Exception.class)
    ResponseEntity<ErrorResponse> failure(Exception e) { return ResponseEntity.status(500).body(new ErrorResponse("The request could not be processed")); }
}
