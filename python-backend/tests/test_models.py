import unittest

from pydantic import ValidationError

from app.models import CreateCollectionRequest, FindSourcesToolArgs, ReadSourceToolArgs


class ModelTests(unittest.TestCase):
    def test_collection_request_accepts_name_and_api_handles_blank_value(self):
        self.assertEqual(CreateCollectionRequest(name="  ").name, "  ")

    def test_read_source_args_require_a_positive_source_number_and_forbid_extra_fields(self):
        self.assertEqual(ReadSourceToolArgs(source_number=1).source_number, 1)
        for payload in ({"source_number": 0}, {"source_number": -1}, {"source_number": 1, "extra": True}):
            with self.subTest(payload=payload), self.assertRaises(ValidationError):
                ReadSourceToolArgs.model_validate(payload)

    def test_find_sources_args_require_nonempty_keyword_and_forbid_extra_fields(self):
        self.assertEqual(FindSourcesToolArgs(keyword="retention").keyword, "retention")
        for payload in ({"keyword": ""}, {"keyword": "retention", "extra": True}):
            with self.subTest(payload=payload), self.assertRaises(ValidationError):
                FindSourcesToolArgs.model_validate(payload)


if __name__ == "__main__":
    unittest.main()
