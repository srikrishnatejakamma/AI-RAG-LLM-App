import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from app import api
from app.ai import EmbeddingService, OpenAIAgentOrchestrator
from app.config import Settings
from app.domain import ChunkData, CollectionData, DocumentData
from app.store import MemoryStore
from app.text_pipeline import checksum, embed_local


class ApiTests(unittest.TestCase):
    def setUp(self):
        self.original_settings = api.ctx.settings
        self.original_store = api.ctx.store
        self.original_sessions = api.ctx.sessions
        self.original_embeddings = api.ctx.embedding_service
        self.original_agent = api.ctx.agent
        self.original_file_search = api.ctx.file_search

        api.ctx.settings = Settings(
            admin_username="admin",
            admin_password="admin-secret",
            editor_username="editor",
            editor_password="editor-secret",
            viewer_username="viewer",
            viewer_password="viewer-secret",
            max_upload_bytes=1024,
            chunk_size=200,
            chunk_overlap=20,
            embedding_provider="local",
            document_parser_provider="native",
            answer_provider="auto",
            openai_api_key="",
        )
        api.ctx.store = MemoryStore()
        api.ctx.sessions = {}
        api.ctx.embedding_service = EmbeddingService(api.ctx.settings)
        api.ctx.agent = OpenAIAgentOrchestrator(api.ctx.settings)
        from app.file_search import OpenAIFileSearch
        api.ctx.file_search = OpenAIFileSearch(api.ctx.settings)
        self.client = TestClient(api.app, raise_server_exceptions=False)

    def tearDown(self):
        api.ctx.settings = self.original_settings
        api.ctx.store = self.original_store
        api.ctx.sessions = self.original_sessions
        api.ctx.embedding_service = self.original_embeddings
        api.ctx.agent = self.original_agent
        api.ctx.file_search = self.original_file_search

    def csrf(self):
        return self.client.get("/api/csrf").json()["token"]

    def login(self, username="admin", password="admin-secret"):
        token = self.csrf()
        return self.client.post(
            "/login",
            data={"username": username, "password": password, "_csrf": token},
            headers={"X-XSRF-TOKEN": token},
        )

    def write_headers(self):
        return {"X-XSRF-TOKEN": self.client.cookies.get("XSRF-TOKEN", "")}

    def test_health_and_csrf_are_available_without_authentication(self):
        health = self.client.get("/api/health")
        csrf = self.client.get("/api/csrf")

        self.assertEqual(health.status_code, 200)
        expected_status = "ok" if api.ctx.settings.openai_api_key else "degraded"
        self.assertEqual(health.json()["status"], expected_status)
        self.assertEqual(csrf.status_code, 200)
        self.assertEqual(csrf.json()["headerName"], "X-XSRF-TOKEN")
        self.assertTrue(csrf.json()["token"])
        self.assertEqual(self.client.cookies.get("XSRF-TOKEN"), csrf.json()["token"])

    def test_openapi_and_swagger_docs_expose_parity_routes_and_session_security(self):
        docs = self.client.get("/swagger-ui.html")
        schema = self.client.get("/v3/api-docs").json()

        self.assertEqual(docs.status_code, 200)
        self.assertEqual(schema["info"]["title"], "Gather RAG API")
        self.assertIn("sessionCookie", schema["components"]["securitySchemes"])
        self.assertIn("csrfToken", schema["components"]["securitySchemes"])
        self.assertIn("/api/collections/{collection_id}/chat", schema["paths"])
        self.assertIn("ChatResponse", schema["components"]["schemas"])
        chat = schema["paths"]["/api/collections/{collection_id}/chat"]["post"]
        self.assertEqual(chat["security"], [{"sessionCookie": [], "csrfToken": []}])
        self.assertNotIn("security", schema["paths"]["/api/health"]["get"])
        self.assertIn("204", schema["paths"]["/login"]["post"]["responses"])
        login_body_ref = schema["paths"]["/login"]["post"]["requestBody"]["content"]["application/x-www-form-urlencoded"]["schema"]["$ref"]
        login_body_name = login_body_ref.rsplit("/", 1)[-1]
        self.assertIn("_csrf", schema["components"]["schemas"][login_body_name]["properties"])
        self.assertEqual(
            set(schema["components"]["schemas"]["ChatResponse"]["properties"]),
            {"answer", "sources", "grounded"},
        )
        upload = schema["paths"]["/api/collections/{collection_id}/documents"]["post"]["responses"]
        self.assertIn("202", upload)
        self.assertIn("503", upload)
        delete = schema["paths"]["/api/collections/{collection_id}"]["delete"]["responses"]
        self.assertIn("204", delete)

    def test_login_requires_csrf_and_rejects_bad_credentials(self):
        response = self.client.post("/login", data={"username": "admin", "password": "wrong"})
        self.assertEqual(response.status_code, 403)

        token = self.csrf()
        response = self.client.post(
            "/login",
            data={"username": "admin", "password": "wrong", "_csrf": token},
            headers={"X-XSRF-TOKEN": token},
        )

        self.assertEqual(response.status_code, 401)
        self.assertEqual(api.ctx.store.audit_events[0].action, "AUTHENTICATION_FAILED")

    def test_login_me_and_logout_lifecycle(self):
        response = self.login()
        self.assertEqual(response.status_code, 204)
        self.assertTrue(self.client.cookies.get("JSESSIONID"))

        me = self.client.get("/api/me")
        self.assertEqual(me.status_code, 200)
        self.assertEqual(me.json(), {"username": "admin", "roles": ["ROLE_ADMIN"]})

        logout = self.client.post("/logout", headers=self.write_headers())
        self.assertEqual(logout.status_code, 204)
        self.assertEqual(self.client.get("/api/me").status_code, 401)
        self.assertIn("LOGOUT", [event.action for event in api.ctx.store.audit_events])

    def test_invalid_csrf_token_blocks_write_even_when_authenticated(self):
        self.assertEqual(self.login().status_code, 204)

        response = self.client.post("/api/collections", json={"name": "Research"}, headers={"X-XSRF-TOKEN": "wrong"})

        self.assertEqual(response.status_code, 403)
        self.assertEqual(response.json()["error"], "CSRF token is invalid")

    def test_create_collection_validates_name_and_trims_valid_name(self):
        self.login()
        empty = self.client.post("/api/collections", json={"name": "  "}, headers=self.write_headers())
        too_long = self.client.post("/api/collections", json={"name": "x" * 81}, headers=self.write_headers())
        created = self.client.post("/api/collections", json={"name": "  Research  "}, headers=self.write_headers())

        self.assertEqual(empty.status_code, 400)
        self.assertEqual(too_long.status_code, 400)
        self.assertEqual(created.status_code, 200)
        self.assertEqual(created.json()["name"], "Research")
        self.assertEqual(created.json()["documentCount"], 0)

    def test_roles_restrict_collection_creation_and_listing_requires_auth(self):
        self.assertEqual(self.client.get("/api/collections").status_code, 401)
        self.assertEqual(self.client.post("/api/collections/00000000-0000-0000-0000-000000000001/chat", json={"question": "question"}).status_code, 401)
        self.login("viewer", "viewer-secret")

        response = self.client.post("/api/collections", json={"name": "Research"}, headers=self.write_headers())

        self.assertEqual(response.status_code, 403)
        self.assertEqual(self.client.get("/api/collections").status_code, 200)

    def test_collection_delete_validates_uuid_existence_and_role(self):
        self.login()
        invalid = self.client.delete("/api/collections/not-a-uuid", headers=self.write_headers())
        missing = self.client.delete("/api/collections/00000000-0000-0000-0000-000000000001", headers=self.write_headers())
        collection = self.client.post("/api/collections", json={"name": "Research"}, headers=self.write_headers()).json()
        deleted = self.client.delete(f"/api/collections/{collection['id']}", headers=self.write_headers())

        self.assertEqual(invalid.status_code, 400)
        self.assertEqual(missing.status_code, 404)
        self.assertEqual(deleted.status_code, 204)
        self.assertEqual(api.ctx.store.audit_events[0].action, "COLLECTION_DELETED")

    def test_document_upload_rejects_empty_oversized_unsupported_and_mismatched_files(self):
        self.login()
        store = api.ctx.store
        created = self.client.post("/api/collections", json={"name": "Research"}, headers=self.write_headers()).json()
        url = f"/api/collections/{created['id']}/documents"

        empty = self.client.post(url, files={"file": ("empty.txt", b"", "text/plain")}, headers=self.write_headers())
        oversized = self.client.post(url, files={"file": ("large.txt", b"x" * 1025, "text/plain")}, headers=self.write_headers())
        unsupported = self.client.post(url, files={"file": ("image.png", b"data", "image/png")}, headers=self.write_headers())
        mismatch = self.client.post(url, files={"file": ("notes.pdf", b"not a pdf", "application/pdf")}, headers=self.write_headers())
        docx_mismatch = self.client.post(url, files={"file": ("notes.docx", b"not a docx", "application/vnd.openxmlformats-officedocument.wordprocessingml.document")}, headers=self.write_headers())

        self.assertEqual(empty.status_code, 400)
        self.assertEqual(oversized.status_code, 400)
        self.assertEqual(unsupported.status_code, 400)
        self.assertEqual(mismatch.status_code, 400)
        self.assertEqual(docx_mismatch.status_code, 400)
        self.assertEqual(len(store.collections[created["id"]].documents), 0)

    def test_txt_upload_sanitizes_filename_and_deduplicates_same_content(self):
        self.login()
        collection = self.client.post("/api/collections", json={"name": "Research"}, headers=self.write_headers()).json()
        url = f"/api/collections/{collection['id']}/documents"
        file = ("../../notes.txt", b"A useful document with a number of searchable words.", "text/plain")

        uploaded = self.client.post(url, files={"file": file}, headers=self.write_headers())
        duplicate = self.client.post(url, files={"file": file}, headers=self.write_headers())

        self.assertEqual(uploaded.status_code, 202)
        self.assertEqual(uploaded.json()["name"], "notes.txt")
        self.assertIn(uploaded.json()["status"], {"PROCESSING", "READY"})
        self.assertEqual(duplicate.status_code, 200)
        self.assertEqual(duplicate.json()["id"], uploaded.json()["id"])
        self.assertEqual(len(api.ctx.store.collections[collection["id"]].documents), 1)

    def test_upload_retries_a_previously_failed_duplicate(self):
        self.login()
        collection = self.client.post("/api/collections", json={"name": "Research"}, headers=self.write_headers()).json()
        payload = b"Retry this readable document after a processing failure."
        failed = DocumentData("failed-doc", "retry.txt", checksum(payload), api.utc_now(), "FAILED", "previous failure")
        api.ctx.store.collections[collection["id"]].documents[failed.id] = failed

        response = self.client.post(
            f"/api/collections/{collection['id']}/documents",
            files={"file": ("retry.txt", payload, "text/plain")},
            headers=self.write_headers(),
        )

        self.assertEqual(response.status_code, 202)
        self.assertEqual(response.json()["id"], failed.id)
        self.assertIn(response.json()["status"], {"PROCESSING", "READY"})
        self.assertIsNone(response.json()["error"])

    def test_document_listing_and_deletion_validate_ids_and_missing_documents(self):
        self.login()
        invalid = self.client.get("/api/collections/not-a-uuid/documents")
        unknown_collection = self.client.get("/api/collections/00000000-0000-0000-0000-000000000001/documents")
        created = self.client.post("/api/collections", json={"name": "Research"}, headers=self.write_headers()).json()
        missing_doc = self.client.delete(
            f"/api/collections/{created['id']}/documents/00000000-0000-0000-0000-000000000002",
            headers=self.write_headers(),
        )

        self.assertEqual(invalid.status_code, 400)
        self.assertEqual(unknown_collection.status_code, 404)
        self.assertEqual(missing_doc.status_code, 404)
        self.assertEqual(self.client.get(f"/api/collections/{created['id']}/documents").json(), [])

    def test_editor_can_delete_document_but_viewer_cannot(self):
        self.login("editor", "editor-secret")
        collection = self.client.post("/api/collections", json={"name": "Research"}, headers=self.write_headers()).json()
        document = DocumentData("00000000-0000-0000-0000-000000000003", "notes.txt", "sum", api.utc_now(), "READY", None)
        api.ctx.store.collections[collection["id"]].documents[document.id] = document

        deleted = self.client.delete(
            f"/api/collections/{collection['id']}/documents/{document.id}", headers=self.write_headers()
        )
        self.assertEqual(deleted.status_code, 204)

        self.client.cookies.clear()
        self.login("viewer", "viewer-secret")
        document = DocumentData("00000000-0000-0000-0000-000000000004", "notes.txt", "sum2", api.utc_now(), "READY", None)
        api.ctx.store.collections[collection["id"]].documents[document.id] = document
        denied = self.client.delete(
            f"/api/collections/{collection['id']}/documents/{document.id}", headers=self.write_headers()
        )
        self.assertEqual(denied.status_code, 403)
        self.assertIn(document.id, api.ctx.store.collections[collection["id"]].documents)

    def test_chat_validates_request_and_reports_empty_or_processing_collection(self):
        self.login()
        invalid_id = self.client.post("/api/collections/not-a-uuid/chat", json={"question": "question"}, headers=self.write_headers())
        created = self.client.post("/api/collections", json={"name": "Research"}, headers=self.write_headers()).json()
        url = f"/api/collections/{created['id']}/chat"
        blank = self.client.post(url, json={"question": "  "}, headers=self.write_headers())
        too_long = self.client.post(url, json={"question": "x" * 2001}, headers=self.write_headers())
        no_evidence = self.client.post(url, json={"question": "What is here?"}, headers=self.write_headers())

        self.assertEqual(invalid_id.status_code, 400)
        self.assertEqual(blank.status_code, 400)
        self.assertEqual(too_long.status_code, 400)
        self.assertFalse(no_evidence.json()["grounded"])
        self.assertEqual(no_evidence.json()["sources"], [])

        doc = DocumentData("doc", "pending.txt", "sum", api.utc_now(), "PROCESSING", None)
        api.ctx.store.collections[created["id"]].documents[doc.id] = doc
        processing = self.client.post(url, json={"question": "What is here?"}, headers=self.write_headers()).json()
        self.assertIn("still being processed", processing["answer"])

    def test_chat_requires_csrf_for_authenticated_write_requests(self):
        self.login()
        collection = self.client.post("/api/collections", json={"name": "Research"}, headers=self.write_headers()).json()

        response = self.client.post(
            f"/api/collections/{collection['id']}/chat",
            json={"question": "retention policy"},
        )

        self.assertEqual(response.status_code, 403)
        self.assertEqual(response.json()["error"], "CSRF token is invalid")

    def test_chat_does_not_fill_results_with_chunks_below_similarity_threshold(self):
        self.login()
        collection = self.client.post("/api/collections", json={"name": "Research"}, headers=self.write_headers()).json()
        question = "What is the retention policy?"
        query_vector = embed_local(question)
        unrelated_vector = [-value for value in query_vector]
        document = DocumentData("doc", "unrelated.txt", "sum", api.utc_now(), "READY", None)
        document.chunks = [ChunkData("The office is near the river and has a garden.", 1, None, unrelated_vector)]
        api.ctx.store.collections[collection["id"]].documents[document.id] = document

        response = self.client.post(f"/api/collections/{collection['id']}/chat", json={"question": question}, headers=self.write_headers())

        self.assertFalse(response.json()["grounded"], response.json())
        self.assertEqual(response.json()["sources"], [])

    def test_chat_returns_grounded_answer_and_bounded_citation(self):
        self.login()
        collection = self.client.post("/api/collections", json={"name": "Research"}, headers=self.write_headers()).json()
        source_text = "The retention policy requires teams to review records every year. " + ("Additional evidence. " * 20)
        document = DocumentData("doc", "policy.txt", "sum", api.utc_now(), "READY", None)
        document.chunks = [ChunkData(source_text, 1, 7, embed_local("retention policy"))]
        api.ctx.store.collections[collection["id"]].documents[document.id] = document

        response = self.client.post(f"/api/collections/{collection['id']}/chat", json={"question": "retention policy"}, headers=self.write_headers())

        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json()["grounded"])
        self.assertEqual(response.json()["sources"][0]["page"], 7)
        self.assertLessEqual(len(response.json()["sources"][0]["excerpt"]), 220)

    def test_chat_retries_extractive_search_over_collection_when_retrieval_misses_evidence(self):
        self.login()
        collection = self.client.post("/api/collections", json={"name": "Research"}, headers=self.write_headers()).json()
        missed = ChunkData("The office is near the river and has a garden.", 1, 1, embed_local("unrelated"))
        relevant = ChunkData("The retention policy requires teams to review records every year.", 2, 2, embed_local("retention policy"))
        document = DocumentData("doc", "policy.txt", "sum", api.utc_now(), "READY", None)
        document.chunks = [missed, relevant]
        api.ctx.store.collections[collection["id"]].documents[document.id] = document

        with patch.object(api.ctx.retriever, "retrieve", return_value=[(document, missed, 0.5)]):
            response = self.client.post(
                f"/api/collections/{collection['id']}/chat",
                json={"question": "retention policy"},
                headers=self.write_headers(),
            )

        self.assertTrue(response.json()["grounded"])
        self.assertIn("review records every year", response.json()["answer"])
        self.assertEqual([source["chunk"] for source in response.json()["sources"]], [2])

    def test_chat_does_not_attach_irrelevant_citations_to_an_unanswered_question(self):
        self.login()
        collection = self.client.post("/api/collections", json={"name": "Research"}, headers=self.write_headers()).json()
        chunk = ChunkData("The office is near the river and has a garden.", 1, 1, embed_local("retention policy"))
        document = DocumentData("doc", "unrelated.txt", "sum", api.utc_now(), "READY", None)
        document.chunks = [chunk]
        api.ctx.store.collections[collection["id"]].documents[document.id] = document

        with patch.object(api.ctx.retriever, "retrieve", return_value=[(document, chunk, 0.5)]):
            response = self.client.post(
                f"/api/collections/{collection['id']}/chat",
                json={"question": "retention policy"},
                headers=self.write_headers(),
            )

        self.assertFalse(response.json()["grounded"], response.json())
        self.assertEqual(response.json()["sources"], [])

    def test_unexpected_chat_failure_hides_internal_error_details(self):
        self.login()
        collection = self.client.post("/api/collections", json={"name": "Research"}, headers=self.write_headers()).json()

        with patch.object(api.ctx.embedding_service, "embed", side_effect=RuntimeError("secret provider detail")):
            response = self.client.post(f"/api/collections/{collection['id']}/chat", json={"question": "question"}, headers=self.write_headers())

        self.assertEqual(response.status_code, 500)
        self.assertEqual(response.json(), {"error": "The request could not be processed"})

    def test_audit_is_admin_only_and_paginates(self):
        self.login("viewer", "viewer-secret")
        denied = self.client.get("/api/admin/audit")
        self.assertEqual(denied.status_code, 403)

        self.client.cookies.clear()
        api.ctx.store.audit_events.clear()
        api.ctx.store.audit_seq = 0
        self.login()
        for index in range(2):
            self.client.post("/api/collections", json={"name": f"Collection {index}"}, headers=self.write_headers())
        first = self.client.get("/api/admin/audit?limit=2")
        self.assertEqual(first.status_code, 200)
        self.assertEqual(len(first.json()["items"]), 2)
        cursor = first.json()["nextCursor"]
        second = self.client.get(f"/api/admin/audit?limit=2&beforeEventId={cursor}")
        self.assertEqual(len(second.json()["items"]), 1)


if __name__ == "__main__":
    unittest.main()
