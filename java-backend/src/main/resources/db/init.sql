CREATE EXTENSION IF NOT EXISTS vector;
CREATE TABLE IF NOT EXISTS system_settings (
  setting_key TEXT PRIMARY KEY,
  setting_value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS audit_events (
  event_id BIGSERIAL PRIMARY KEY,
  occurred_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  actor VARCHAR(120) NOT NULL,
  action VARCHAR(80) NOT NULL,
  resource_type VARCHAR(40) NOT NULL,
  resource_id VARCHAR(80),
  outcome VARCHAR(40) NOT NULL,
  request_id VARCHAR(64)
);
CREATE INDEX IF NOT EXISTS audit_events_time_idx ON audit_events(occurred_at DESC);
CREATE TABLE IF NOT EXISTS collections (
  id UUID PRIMARY KEY,
  name VARCHAR(80) NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE TABLE IF NOT EXISTS documents (
  id UUID PRIMARY KEY,
  collection_id UUID NOT NULL REFERENCES collections(id) ON DELETE CASCADE,
  name TEXT NOT NULL,
  sha256 CHAR(64) NOT NULL,
  status VARCHAR(16) NOT NULL DEFAULT 'PROCESSING' CHECK (status IN ('PROCESSING','READY','FAILED')),
  error_message TEXT,
  uploaded_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  openai_file_id TEXT,
  UNIQUE(collection_id, sha256)
);
ALTER TABLE documents ADD COLUMN IF NOT EXISTS openai_file_id TEXT;
CREATE TABLE IF NOT EXISTS chunks (
  id BIGSERIAL PRIMARY KEY,
  collection_id UUID NOT NULL REFERENCES collections(id) ON DELETE CASCADE,
  document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
  chunk_index INTEGER NOT NULL,
  page_number INTEGER,
  content TEXT NOT NULL,
  embedding vector(1536) NOT NULL,
  UNIQUE(document_id, chunk_index)
);
CREATE INDEX IF NOT EXISTS chunks_embedding_idx ON chunks USING hnsw (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS chunks_collection_idx ON chunks(collection_id);
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS page_number INTEGER;
