import unittest
from unittest.mock import call, patch

from app.config import Settings
from app.file_search import OpenAIFileSearch


class OpenAIFileSearchTests(unittest.TestCase):
    def service(self):
        return OpenAIFileSearch(Settings(
            admin_password="secret",
            openai_api_key="test-key",
            openai_vector_store_id="vs_test",
            openai_base_url="https://api.openai.com/v1",
        ))

    def test_upload_attaches_collection_metadata_and_waits_for_indexing(self):
        service = self.service()
        with patch.object(service, "_request", side_effect=[
            {"id": "file_1"}, {"id": "vsf_1"}, {"status": "in_progress"}, {"status": "completed"}
        ]) as request, patch("app.file_search.time.sleep"):
            file_id = service.upload_and_index("collection-1", "document-1", "policy.txt", b"policy")

        self.assertEqual(file_id, "file_1")
        self.assertEqual(request.call_args_list[1], call(
            "POST", "/vector_stores/vs_test/files",
            json={"file_id": "file_1", "attributes": {"collection_id": "collection-1", "document_id": "document-1"}},
        ))

    def test_search_filters_to_collection_and_returns_file_results_for_citations(self):
        service = self.service()
        response = {
            "output_text": "Retention lasts seven years.",
            "output": [{"type": "file_search_call", "results": [{"file_id": "file_1", "content": [{"type": "text", "text": "Retain records for seven years."}]}]}],
        }
        with patch.object(service, "_request", return_value=response) as request:
            answer, results = service.answer("How long?", "collection-1")

        self.assertEqual(answer, "Retention lasts seven years.")
        self.assertEqual(results[0]["file_id"], "file_1")
        payload = request.call_args.kwargs["json"]
        self.assertEqual(payload["tools"][0]["filters"], {"type": "eq", "key": "collection_id", "value": "collection-1"})
        self.assertIn("file_search_call.results", payload["include"])
        self.assertEqual(payload["tools"][0]["max_num_results"], 10)


if __name__ == "__main__":
    unittest.main()
