import unittest
from types import SimpleNamespace

from app.ai import (
    EmbeddingService,
    FunctionTool,
    OpenAIAgentOrchestrator,
    _extract_sentences,
    contextualize_question,
    correct_query_spelling,
    extractive_answer,
    extract_temporal_facts,
    is_collection_overview,
)
from app.config import Settings
from app.models import ReadSourceToolArgs


class AiTests(unittest.TestCase):
    def test_temporal_request_collects_dates_and_periods_across_passages(self):
        passages = [
            "[handbook.pdf, chunk 1, page 4]\nSubmit your request by March 15, 2026.",
            "[handbook.pdf, chunk 2, page 8]\nPlease give at least [two weeks] notice before resigning.",
            "[handbook.pdf, chunk 3, page 12]\nEmployees should keep their contact information current.",
        ]

        facts = extract_temporal_facts("Find important dates and deadlines", passages)

        self.assertEqual(len(facts), 2)
        self.assertIn("March 15, 2026", facts[0][0])
        self.assertIn("two weeks", facts[1][0])
        self.assertEqual([index for _, index in facts], [0, 1])

    def test_temporal_extraction_does_not_run_for_non_temporal_requests(self):
        passages = ["[handbook.pdf, page 1] Submit your request by March 15, 2026."]

        self.assertEqual(extract_temporal_facts("Summarize this document", passages), [])

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
            overview=True,
        )

        self.assertIn("Teams review records annually", answer)
        self.assertIn("Managers approve retention exceptions", answer)

    def test_overview_does_not_present_unfilled_template_values_as_facts(self):
        answer = extractive_answer(
            "Summarize the key points",
            [
                "[handbook.txt] Full-time employees work at least [30 hours] per week on average. "
                "Managers must review team access permissions every quarter. "
                "Employees must report suspected security incidents to the security team promptly."
            ],
            overview=True,
        )

        self.assertIn("placeholders", answer)
        self.assertNotIn("[30 hours]", answer)
        self.assertIn("review team access permissions", answer)

    def test_policy_overview_summarizes_evidence_and_skips_intro_boilerplate(self):
        answer = extractive_answer(
            "What is the full list of main policies?",
            ["[handbook] Workplace policies. This section describes policies that apply to everyone. "
             "Employees receive overtime pay for approved extra hours. "
             "Annual leave requests need manager approval before time off. "
             "Employees must report workplace safety hazards to their manager promptly. "
             "Employees may report harassment directly to Human Resources. "
             "Staff must record attendance and notify managers about absences."],
            overview=True,
        )

        self.assertIn("Based on the document", answer)
        self.assertIn("overtime pay", answer)
        self.assertIn("safety hazards", answer)
        self.assertNotIn("This section describes", answer)

    def test_policy_overview_uses_document_hierarchy_instead_of_random_sentences(self):
        toc = """[Employee-Handbook.pdf, chunk 2, page 2]
Welcome 4

Employment basics 5
    Employment contract types 5
    Equal opportunity employment 5

Workplace policies 8
    Confidentiality and data protection 8
    Harassment and violence 9

Policy revision 35"""

        answer = extractive_answer("What are the main policies?", [toc], overview=True)

        self.assertIn("Employment basics (page 5)", answer)
        self.assertIn("Workplace policies (page 8)", answer)
        self.assertNotIn("Welcome", answer)
        self.assertNotIn("Policy revision", answer)

    def test_pdf_wrapped_lines_become_complete_sentences_and_instruction_fragments_are_dropped(self):
        passage = """[Employee-Handbook.pdf, chunk 51, page 27]
Paid time off (PTO)
Employees receive [20 days] of Paid Time Off (PTO) per year. Your PTO accrual begins
the day you join our company.

[Insert this if employees are in the U.S: If you are an exempt employee, you are not"""

        sentences = _extract_sentences(passage, ["paid time off (pto)"])

        self.assertIn("Employees receive [20 days] of Paid Time Off (PTO) per year.", sentences)
        self.assertIn("Your PTO accrual begins the day you join our company.", sentences)
        self.assertFalse(any("Insert this" in sentence for sentence in sentences))

    def test_detail_answer_does_not_jump_to_a_different_section_on_one_shared_word(self):
        answer = extractive_answer(
            "Annual Leave",
            [
                "[performance.pdf] Revisit those goals during [annual/ bi-annual/ quarterly] performance reviews.",
                "[parental.pdf] Eligible employees may take up to 12 weeks of parental leave after a birth or adoption.",
                "[pto.pdf] Paid time off requests must be submitted to a supervisor 10 days before leave begins.",
            ],
            overview=False,
        )

        self.assertIn("I don't have enough information", answer)
        self.assertNotIn("parental leave", answer)

    def test_annual_leave_is_a_detail_query_even_when_the_collection_only_says_leave(self):
        passages = [
            "Paid time off (PTO) requests need supervisor approval.",
            "Eligible employees may take parental leave after a birth or adoption.",
        ]

        self.assertFalse(is_collection_overview("Annual Leave", passages))

    def test_unmatched_two_word_topic_is_not_misclassified_as_a_summary(self):
        self.assertFalse(is_collection_overview("retention policy", ["The office is near the river and has a garden."]))

    def test_extractive_answer_reports_no_supporting_evidence(self):
        answer = extractive_answer("What is the retention policy?", ["[office.txt] The office is near the river."])

        self.assertIn("I don't have enough information", answer)

    def test_lexical_matching_handles_regular_plural_forms(self):
        answer = extractive_answer(
            "What is the minimum password length?",
            ["[passwords.txt] Account passwords must contain at least 14 characters."],
            overview=False,
        )

        self.assertIn("14 characters", answer)

    def test_query_correction_uses_collection_terms_and_preserves_short_words(self):
        passages = ["Annual leave requests must be submitted to a supervisor."]

        self.assertEqual(correct_query_spelling("ask for leavs", passages), "ask for leave")

    def test_specific_followup_does_not_inherit_a_collection_overview(self):
        passages = ["Paid time off (PTO) requests need supervisor approval.", "Parental leave lasts 12 weeks."]

        self.assertEqual(
            contextualize_question("Explain PTO", ["What are the main policies?"], passages),
            "Explain PTO",
        )
        self.assertEqual(
            contextualize_question("more", ["What are the main policies?", "Explain PTO"], passages),
            "Explain PTO more",
        )

    def test_followup_can_resolve_topic_from_previous_grounded_answer(self):
        passages = [
            "Paid time off (PTO) requests need approval. Employees may use PTO or sick leave.",
            "Eligible employees may take parental leave after a birth or adoption.",
        ]

        self.assertEqual(
            contextualize_question(
                "Annual Leave",
                ["PTO"],
                passages,
                previous_answers=["PTO can be used for time away from work. Employees may use PTO or sick leave."],
            ),
            "PTO Annual Leave",
        )

    def test_annual_leave_followup_uses_the_previous_pto_evidence(self):
        passages = [
            "[pto.pdf, page 1] Paid time off (PTO)\nEmployees receive [20 days] of Paid Time Off (PTO) per year. Employees may use PTO or sick leave.",
            "[parental.pdf, page 1] Eligible employees may take up to 12 weeks of parental leave after a birth or adoption.",
        ]
        answer_text = "PTO can be used for time away from work. Employees may use PTO or sick leave."
        contextual_question = contextualize_question(
            "Annual Leave", ["PTO"], passages, previous_answers=[answer_text]
        )

        answer = extractive_answer(
            contextual_question, passages, overview=False, section_context=passages
        )

        self.assertIn("Paid Time Off (PTO)", answer)
        self.assertNotIn("parental leave", answer)

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

    def test_extractive_answer_provider_disables_unfunded_remote_calls(self):
        settings = Settings(admin_password="secret", openai_api_key="configured", answer_provider="extractive")
        agent = OpenAIAgentOrchestrator(settings)

        self.assertFalse(agent.available())
        self.assertEqual(settings.answer_provider_health()["answerProvider"], "extractive")


if __name__ == "__main__":
    unittest.main()
