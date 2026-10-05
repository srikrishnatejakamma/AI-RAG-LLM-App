# Gather — Agentic RAG Generator

Gather is a collection based document question answering workspace. Upload PDF, DOCX, or TXT files, then ask questions grounded in the selected collection. Answers can include source passages and PDF page references.

## Repository layout

- `java-backend/` — Spring Boot API, storage adapters, and Java tests.
- `frontend/` — React and Vite user interface.
- `python-backend/` — optional FastAPI implementation and Python tests.
- `docs/` — architecture and operating notes.

## Run the Java application

Requirements: Java 21 and Gradle 8.8 or later.

From the repository root:

```powershell
gradle :java-backend:bootRun
```

Open <http://localhost:8080> for the built-in UI. When `RAG_ADMIN_PASSWORD` is not set, startup generates a one-time admin password and prints it in the application log. To choose a password, set it before starting the service:

```powershell
$env:RAG_ADMIN_PASSWORD = "choose-a-local-password"
gradle :java-backend:bootRun
```

The default storage is in memory, so collections are cleared when the Java process stops.

### Run the React UI in development

Start the Java application first, then use another terminal:

```powershell
cd frontend
npm ci
npm run dev
```

Open <http://localhost:5173/react/>. The Vite development server proxies API, login, and logout requests to the Java application on port 8080.

To build the React UI into the Java application:

```powershell
cd frontend
npm ci
npm run build
cd ..
gradle :java-backend:bootJar
```

The packaged UI is served from `/react/`; the built-in UI remains at `/`.

## Optional Python API

The FastAPI implementation is a separate, in-memory alternative to the Java service. Run one backend at a time on port 8080. From the repository root:

```powershell
cd python-backend
python -m pip install -r requirements.txt
python -m uvicorn main:app --host 0.0.0.0 --port 8080 --reload
```

Its interactive API documentation is at <http://localhost:8080/swagger-ui.html>, OpenAPI JSON at <http://localhost:8080/v3/api-docs>, and ReDoc at <http://localhost:8080/redoc>. The React development proxy targets Java, so use the Python API directly unless you change the Vite proxy target.

The Python API combines local dense embeddings with a cached scikit-learn TF-IDF index for lexical retrieval. OpenAI features are optional and require `OPENAI_API_KEY`. Hosted File Search also requires `OPENAI_VECTOR_STORE_ID` and `RAG_RETRIEVAL_PROVIDER=openai-file-search`. See [architecture.md](docs/architecture.md) for provider details and design limits.

For semantic local embeddings in the Python API, install its optional Hugging Face dependencies and select the Transformers provider:

```powershell
cd python-backend
python -m pip install -r requirements-ml.txt
$env:RAG_EMBEDDING_PROVIDER = "transformers"
# Optional: choose a different Hugging Face encoder model.
$env:RAG_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
python -m uvicorn main:app --host 0.0.0.0 --port 8080 --reload
```

The model is downloaded on first use and cached by Hugging Face. This provider uses normalized mean-pooled Transformer embeddings; its vectors are kept in memory with this Python backend's in-memory documents. Keep one model configured for the lifetime of the process, and re-upload documents after changing models. The default `local` provider needs no model download or ML dependencies.

## Run with PostgreSQL

Docker Compose builds the Java application and React UI and starts PostgreSQL with pgvector. Set passwords before starting:

```powershell
$env:POSTGRES_PASSWORD = "choose-a-database-password"
$env:RAG_ADMIN_PASSWORD = "choose-an-admin-password"
docker compose up --build
```

The application is available at <http://localhost:8080>. Compose uses PostgreSQL storage, which persists collections across application restarts. To use OpenAI embeddings or answer generation, also set `OPENAI_API_KEY`; hosted File Search additionally requires `OPENAI_VECTOR_STORE_ID` and `RAG_RETRIEVAL_PROVIDER=openai-file-search`.

## API and access

The API is documented at `/swagger-ui.html`; its OpenAPI JSON is at `/v3/api-docs`.

- `GET /api/health` — readiness and provider status.
- `GET /api/csrf` — CSRF token for browser sessions.
- `POST /login` and `POST /logout` — session authentication.
- `/api/collections` — create, list, and delete collections.
- `/api/collections/{id}/documents` — list and upload documents.
- `/api/collections/{id}/documents/{documentId}` — remove a document.
- `POST /api/collections/{id}/chat` — ask a question about a collection.
- `GET /api/admin/audit` — paginated audit events for administrators.

Java accounts are configured through `RAG_ADMIN_USERNAME` / `RAG_ADMIN_PASSWORD`, `RAG_EDITOR_USERNAME` / `RAG_EDITOR_PASSWORD`, and `RAG_VIEWER_USERNAME` / `RAG_VIEWER_PASSWORD`. State-changing browser requests require a CSRF token. Set `SESSION_COOKIE_SECURE=true` when serving the application over HTTPS.
