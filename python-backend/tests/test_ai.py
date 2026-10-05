import unittest
from types import SimpleNamespace

from app.ai import EmbeddingService, FunctionTool, OpenAIAgentOrchestrator, extractive_answer
from app.config import Settings
from app.models import ReadSourceToolArgs


class AiTests(unittest.TestCase):
    def test_extractive_answer_selects_relevant_evidence(self):
        answer = extractive_answer(
            "What is the retention policy?",
            [
                "[policy.txt] The retention policy requires annual review of records.",
                "[office.txt] The office is near the river and has a large garden.",
            ],
        )

        self.assertIn("retention policy requires annual review", answer)
        self.assertNotIn("office is near the river", answer)

    def test_extractive_summary_returns_evidence_even_without_keyword_overlap(self):
        answer = extractive_answer(
            "Summarize the key points",
            ["[overview.txt] Teams review records annually. Managers approve retention exceptions."],
        )

        self.assertIn("Teams review records annually", answer)
        self.assertIn("Managers approve retention exceptions", answer)

    def test_full_policy_list_returns_multiple_policy_areas_and_skips_intro_boilerplate(self):
        answer = extractive_answer(
            "What is the full list of main policies?",
            ["[handbook] Workplace policies. This section describes policies that apply to everyone. "
             "Employees receive overtime pay for approved extra hours. "
             "Annual leave requests need manager approval before time off. "
             "Employees must report workplace safety hazards to their manager promptly. "
             "Employees may report harassment directly to Human Resources. "
             "Staff must record attendance and notify managers about absences."],
        )

        self.assertIn("overtime pay", answer)
        self.assertIn("Annual leave", answer)
        self.assertIn("safety hazards", answer)
        self.assertIn("report harassment", answer)
        self.assertIn("record attendance", answer)
        self.assertNotIn("This section describes", answer)

    def test_extractive_answer_reports_no_supporting_evidence(self):
        answer = extractive_answer("What is the retention policy?", ["[office.txt] The office is near the river."])

        self.assertIn("I don't have enough information", answer)

    def test_embedding_service_uses_local_provider_by_default_and_rejects_missing_openai_key(self):
        local = EmbeddingService(Settings(admin_password="secret", embedding_provider="local", openai_api_key=""))
        openai = EmbeddingService(Settings(admin_password="secret", embedding_provider="openai", openai_api_key=""))

        self.assertEqual(len(local.embed("retention policy")), 1536)
        self.assertEqual(local.embed_many(["retention policy", "annual review"]), [local.embed("retention policy"), local.embed("annual review")])
        with self.assertRaisesRegex(RuntimeError, "OPENAI_API_KEY is required"):
            openai.embed("retention policy")

    def test_openai_embedding_batches_inputs_and_restores_provider_order(self):
        service = EmbeddingService(Settings(admin_password="secret", embedding_provider="openai", openai_api_key="configured"))
        requests = []

        class Embeddings:
            def create(self, **kwargs):
                start = sum(len(batch) for batch in requests)
                requests.append(kwargs["input"])
                return SimpleNamespace(data=[
                    SimpleNamespace(index=index, embedding=[float(start + index)] * 1536)
                    for index in reversed(range(len(kwargs["input"])))
                ])

        service.client = SimpleNamespace(embeddings=Embeddings())
        vectors = service.embed_many([f"text-{index}" for index in range(65)])

        self.assertEqual([len(batch) for batch in requests], [64, 1])
        self.assertEqual(len(vectors), 65)
        self.assertEqual(vectors[0][0], 0.0)
        self.assertEqual(vectors[64][0], 64.0)

    def test_agent_tools_read_sources_and_find_keywords(self):
        agent = OpenAIAgentOrchestrator(Settings(admin_password="secret", openai_api_key=""))
        passages = ["[policy.txt] Records follow a seven year retention policy.", "[other.txt] Office hours are nine to five."]

        self.assertIn("seven year retention", agent._read_source_tool(ReadSourceToolArgs(source_number=1), passages))
        self.assertEqual(agent._read_source_tool(ReadSourceToolArgs(source_number=3), passages), "Source number out of range. Valid range is 1..2.")
        self.assertIn("[Source 1] Records follow", agent._find_sources_tool(agent.tool_map["find_sources"].args_model(keyword="RETENTION"), passages))
        self.assertEqual(agent._find_sources_tool(agent.tool_map["find_sources"].args_model(keyword="missing"), passages), "No matching source passages found.")

    def test_function_tool_exports_schema_and_agent_unavailable_without_key(self):
        agent = OpenAIAgentOrchestrator(Settings(admin_password="secret", openai_api_key=""))
        tool = FunctionTool("read_source", "Read a source", ReadSourceToolArgs, lambda args, _: str(args.source_number))

        schema = tool.as_openai_tool()
        self.assertFalse(agent.available())
        self.assertEqual(schema["function"]["name"], "read_source")
        self.assertIn("source_number", schema["function"]["parameters"]["properties"])

    def test_agent_is_unavailable_for_invalid_answer_provider(self):
        agent = OpenAIAgentOrchestrator(Settings(admin_password="secret", openai_api_key="configured", answer_provider="invalid"))

        self.assertFalse(agent.available())


if __name__ == "__main__":
    unittest.main()
