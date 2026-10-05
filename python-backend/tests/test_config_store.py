import unittest

from app.config import Settings
from app.store import MemoryStore


class SettingsTests(unittest.TestCase):
    def test_auth_health_and_optional_roles(self):
        settings = Settings(
            admin_username="root",
            admin_password="admin-secret",
            editor_username="editor",
            editor_password="editor-secret",
            viewer_username="viewer",
            viewer_password="viewer-secret",
        )

        self.assertEqual(settings.users["root"], ("admin-secret", ["ROLE_ADMIN"]))
        self.assertEqual(settings.users["editor"], ("editor-secret", ["ROLE_EDITOR"]))
        self.assertEqual(settings.users["viewer"], ("viewer-secret", ["ROLE_VIEWER"]))
        self.assertEqual(settings.auth_health(), {"authOneTimeAdminCredentials": "false", "authAdminUsername": "root"})

    def test_optional_account_requires_both_username_and_password(self):
        with self.assertRaisesRegex(RuntimeError, "Both username and password must be set for the EDITOR account"):
            Settings(admin_password="secret", editor_username="editor")
        with self.assertRaisesRegex(RuntimeError, "Both username and password must be set for the VIEWER account"):
            Settings(admin_password="secret", viewer_password="viewer-secret")

    def test_answer_provider_health_reports_invalid_missing_key_and_configured_key(self):
        invalid = Settings(admin_password="secret", answer_provider="custom").answer_provider_health()
        no_key = Settings(admin_password="secret", answer_provider="legacy").answer_provider_health()
        configured = Settings(admin_password="secret", answer_provider="spring-ai", openai_api_key="key").answer_provider_health()

        self.assertEqual(invalid["answerProviderStatus"], "degraded")
        self.assertIn("must be auto, spring-ai, or legacy", invalid["answerProviderMessage"])
        self.assertEqual(no_key["answerProviderStatus"], "degraded")
        self.assertIn("extractive fallback", no_key["answerProviderMessage"])
        self.assertEqual(configured["answerProviderStatus"], "ok")


class MemoryStoreTests(unittest.IsolatedAsyncioTestCase):
    async def test_audit_ids_are_sequential_events_are_newest_first_and_values_are_bounded(self):
        store = MemoryStore()
        first = await store.record_audit("a" * 130, "CREATE", "collection", "one", "SUCCESS", "request-1")
        second = await store.record_audit("admin", "DELETE", "collection", "one", "SUCCESS", "request-2")

        self.assertEqual((first.event_id, second.event_id), (1, 2))
        self.assertEqual(len(first.actor), 120)
        self.assertEqual([event.event_id for event in store.audit_events], [2, 1])


if __name__ == "__main__":
    unittest.main()
