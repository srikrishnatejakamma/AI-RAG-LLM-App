from __future__ import annotations

import logging
import os
import secrets
import importlib.util
import shutil
from dataclasses import dataclass, field


log = logging.getLogger("python-rag-backend")


@dataclass
class Settings:
    chunk_size: int = int(os.getenv("RAG_CHUNK_SIZE", "1200"))
    chunk_overlap: int = int(os.getenv("RAG_CHUNK_OVERLAP", "180"))
    max_upload_bytes: int = int(os.getenv("RAG_MAX_UPLOAD_BYTES", str(10 * 1024 * 1024)))
    minimum_score: float = float(os.getenv("RAG_RETRIEVAL_MINIMUM_SCORE", "0.04"))
    retrieval_top_k: int = max(3, min(20, int(os.getenv("RAG_RETRIEVAL_TOP_K", "10"))))
    embedding_provider: str = os.getenv("RAG_EMBEDDING_PROVIDER", "local").lower()
    embedding_model: str = os.getenv("RAG_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    retrieval_provider: str = os.getenv("RAG_RETRIEVAL_PROVIDER", "local").lower()
    document_parser_provider: str = os.getenv("RAG_DOCUMENT_PARSER", "auto").lower()
    document_ocr_enabled: bool = os.getenv("RAG_DOCUMENT_OCR_ENABLED", "true").strip().lower() in {"1", "true", "yes", "on"}
    document_ocr_engine: str = os.getenv("RAG_DOCUMENT_OCR_ENGINE", "rapidocr").strip().lower()
    document_ocr_languages: tuple[str, ...] = tuple(
        language.strip()
        for language in os.getenv("RAG_DOCUMENT_OCR_LANGUAGES", "iso:en").split(",")
        if language.strip()
    )
    document_max_pages: int = max(1, int(os.getenv("RAG_DOCUMENT_MAX_PAGES", "300")))
    text_encoding: str = os.getenv("RAG_TEXT_ENCODING", "auto").strip()
    answer_provider: str = os.getenv("RAG_ANSWER_PROVIDER", "auto").lower()
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    openai_base_url: str = os.getenv("OPENAI_BASE_URL", "https://api.openai.com")
    openai_chat_model: str = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
    openai_embedding_model: str = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
    openai_vector_store_id: str = os.getenv("OPENAI_VECTOR_STORE_ID", "")
    admin_username: str = os.getenv("RAG_ADMIN_USERNAME", "admin")
    admin_password: str = os.getenv("RAG_ADMIN_PASSWORD", "")
    editor_username: str = os.getenv("RAG_EDITOR_USERNAME", "")
    editor_password: str = os.getenv("RAG_EDITOR_PASSWORD", "")
    viewer_username: str = os.getenv("RAG_VIEWER_USERNAME", "")
    viewer_password: str = os.getenv("RAG_VIEWER_PASSWORD", "")
    one_time_admin_credentials: bool = False
    users: dict[str, tuple[str, list[str]]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.admin_password:
            self.one_time_admin_credentials = True
            self.admin_password = secrets.token_urlsafe(24)
            log.warning(
                "Generated one-time local admin password. Username: %s Password: %s. "
                "Set RAG_ADMIN_PASSWORD to keep credentials across restarts.",
                self.admin_username,
                self.admin_password,
            )
        self.users[self.admin_username] = (self.admin_password, ["ROLE_ADMIN"])
        if self.editor_username or self.editor_password:
            if not self.editor_username or not self.editor_password:
                raise RuntimeError("Both username and password must be set for the EDITOR account")
            self.users[self.editor_username] = (self.editor_password, ["ROLE_EDITOR"])
        if self.viewer_username or self.viewer_password:
            if not self.viewer_username or not self.viewer_password:
                raise RuntimeError("Both username and password must be set for the VIEWER account")
            self.users[self.viewer_username] = (self.viewer_password, ["ROLE_VIEWER"])

    def auth_health(self) -> dict[str, str]:
        return {
            "authOneTimeAdminCredentials": str(self.one_time_admin_credentials).lower(),
            "authAdminUsername": self.admin_username,
        }

    def answer_provider_health(self) -> dict[str, str]:
        provider = self.answer_provider if self.answer_provider in {"auto", "spring-ai", "legacy"} else "invalid"
        if provider == "invalid":
            return {
                "answerProvider": provider,
                "answerProviderStatus": "degraded",
                "answerProviderMessage": "RAG_ANSWER_PROVIDER must be auto, spring-ai, or legacy.",
            }
        if not self.openai_api_key:
            return {
                "answerProvider": provider,
                "answerProviderStatus": "degraded",
                "answerProviderMessage": "OPENAI_API_KEY is missing; chat will run in extractive fallback mode.",
            }
        return {
            "answerProvider": provider,
            "answerProviderStatus": "ok",
            "answerProviderMessage": "OPENAI_API_KEY is configured. Provider connectivity is verified at runtime.",
        }

    def embedding_provider_name(self) -> str:
        return self.embedding_provider if self.embedding_provider in {"local", "openai", "transformers"} else "invalid"

    def retrieval_provider_health(self) -> dict[str, str]:
        if self.retrieval_provider not in {"local", "openai-file-search"}:
            return {"retrievalProvider": "invalid", "retrievalProviderStatus": "degraded", "retrievalProviderMessage": "RAG_RETRIEVAL_PROVIDER must be local or openai-file-search."}
        if self.retrieval_provider == "openai-file-search":
            if not self.openai_api_key or not self.openai_vector_store_id:
                return {"retrievalProvider": self.retrieval_provider, "retrievalProviderStatus": "degraded", "retrievalProviderMessage": "OPENAI_API_KEY and OPENAI_VECTOR_STORE_ID are required for hosted File Search."}
            return {"retrievalProvider": self.retrieval_provider, "retrievalProviderStatus": "ok", "retrievalProviderMessage": "Hosted File Search is configured; connectivity is checked when used."}
        return {"retrievalProvider": self.retrieval_provider, "retrievalProviderStatus": "ok", "retrievalProviderMessage": "Local retrieval is enabled."}

    def document_parser_health(self) -> dict[str, str]:
        provider = self.document_parser_provider
        if provider not in {"auto", "native", "docling"}:
            return {
                "documentParser": "invalid",
                "documentParserStatus": "degraded",
                "documentParserMessage": "RAG_DOCUMENT_PARSER must be auto, native, or docling.",
            }
        if provider == "native":
            return {
                "documentParser": "native",
                "documentParserStatus": "ok",
                "documentParserMessage": "Native parsing is enabled; OCR and advanced layout analysis are disabled.",
            }
        if importlib.util.find_spec("docling") is None:
            return {
                "documentParser": provider,
                "documentParserStatus": "degraded",
                "documentParserMessage": "Docling is missing; PDF and DOCX use the native fallback without OCR. Install python-backend requirements.",
            }
        if self.document_ocr_enabled:
            if self.document_ocr_engine not in {"rapidocr", "tesseract"}:
                return {
                    "documentParser": provider,
                    "documentParserStatus": "degraded",
                    "documentParserMessage": "RAG_DOCUMENT_OCR_ENGINE must be rapidocr or tesseract.",
                }
            if self.document_ocr_engine == "rapidocr" and importlib.util.find_spec("rapidocr_onnxruntime") is None:
                return {
                    "documentParser": provider,
                    "documentParserStatus": "degraded",
                    "documentParserMessage": "RapidOCR is missing; install Docling with its rapidocr extra.",
                }
            if self.document_ocr_engine == "tesseract" and shutil.which("tesseract") is None:
                return {
                    "documentParser": provider,
                    "documentParserStatus": "degraded",
                    "documentParserMessage": "Tesseract is missing from PATH; install Tesseract or select RapidOCR.",
                }
        return {
            "documentParser": provider,
            "documentParserStatus": "ok",
            "documentParserMessage": "Docling layout and OCR parsing are configured.",
        }
