# Architecture and operating decisions

## Request path

The application is a modular Spring Boot service with a browser UI served from the same origin. Collections scope every document and every retrieval query. Upload requests validate size, extension, and detected file type, calculate a SHA-256 checksum, and return while a bounded worker pool parses, chunks, embeds, and indexes the document. The UI polls for `PROCESSING`, `READY`, and `FAILED` status. Failed documents can be retried by uploading the same file again.

The original static UI remains at `/`. A separate React/Vite app lives in `frontend/`; its development server runs at `http://localhost:5173/react/` and proxies API, login, and logout requests to Spring Boot. A Vite production build is packaged by Gradle or the multi-stage Docker build and served from `/react/` by the backend.

Retrieval returns up to `RAG_RETRIEVAL_TOP_K` chunks (default 10, clamped to 3..20) from the requested collection that meet the configured minimum similarity score. Low-scoring chunks are not added just to fill the result list. PostgreSQL uses pgvector cosine search and HNSW; local mode uses deterministic 1,536-dimensional feature hashing and an in-memory index. PDF chunks preserve page numbers. Chat can call a configurable OpenAI compatible endpoint; without a key it selects concise, question-relevant sentences from retrieved passages. Source citations are returned separately. The prompt treats all file content as untrusted evidence and tells the model to ignore embedded instructions.

## Storage choices

- **Local development:** `RAG_STORAGE=memory`. Fast setup, no Docker or credentials, and data is cleared on restart.
- **Persistent deployment:** `RAG_STORAGE=postgres` with PostgreSQL and pgvector. HikariCP bounds database connections; DB uniqueness makes ingestion idempotent per collection; every vector query includes the collection ID predicate.

The database records the embedding profile used to create vectors. Startup fails when that profile changes while vectors remain, because mixing vectors from different models would quietly degrade retrieval. Remove and re-upload documents before switching providers/models. The vector column and provider are configured for 1,536 dimensions.

## Capacity and failure controls

- Upload size is capped at 10 MB. Tika extraction is capped at two million characters and one document at 1,800 chunks.
- Ingestion runs on two to three workers with a queue of four documents, bounding concurrent parsing and provider requests. Provider calls have connection/read timeouts; embedding requests are batched in groups of 64.
- SHA-256 checksums prevent duplicate rows. A failed checksum match can be submitted again to retry processing.
- Database connections use a 12-connection maximum pool and a five-second acquisition timeout.
- LLM output is capped, retrieval returns at most 20 chunks, and missing evidence produces an explicit no-answer response.
- Collection deletion cascades to files and chunks; document deletion removes its indexed content.

## Security and data isolation

- Filenames are reduced to a basename and are never used as paths.
- The upload extension is checked against Tika's detected type; unsupported types, empty files, and oversized uploads are rejected.
- Database statements are parameterized. Collection IDs are included in retrieval, document deletion, and checksum operations.
- Provider credentials are read from environment variables. Uploaded text is untrusted and cannot override the system prompt.
- Spring Security sessions use BCrypt-backed environment-provisioned accounts, SameSite/HTTP-only cookies, and CSRF tokens. `ADMIN`, `EDITOR`, and `VIEWER` are enforced at the HTTP and method layers.
- Request logs use Logstash JSON with request IDs. Audit records contain actor, action, outcome, resource identifiers, and request ID, but not question text, passwords, or document content. PostgreSQL retains audit events; local memory mode keeps the most recent 10,000.
- `GET /api/admin/audit` is admin-only and cursor-paginated. OpenAPI is generated at `/v3/api-docs`; Swagger UI is at `/swagger-ui.html`.
- Roles are instance-wide: all authenticated users currently see the same collections. Add tenant ownership and an OIDC identity provider with MFA before exposing this to multiple organizations or the public internet. Account provisioning is environment-based; there is no self-service user management or password reset flow.
- Set `SESSION_COOKIE_SECURE=true` behind HTTPS. Authentication throttling and audit retention policy should be configured at the identity provider and deployment boundary for production.

## Trade-offs and next steps

The single service keeps the assessment inspectable and avoids premature service boundaries. The local feature-hash adapter is useful offline, but it is lexical and less accurate for paraphrases than a semantic embedding model. Production deployments should configure semantic embeddings and keep the configured model stable. OCR for image-only PDFs, multi-user authorization, chat history, durable upload blob storage, reindex jobs, audit events, and metrics are sensible additions for regulated or high-volume workloads. The current in-memory mode does not survive restarts; PostgreSQL mode is the persistence path.
