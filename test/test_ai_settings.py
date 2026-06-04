import os
import unittest
from unittest.mock import patch

from ai.settings import get_ai_settings


class AISettingsTests(unittest.TestCase):
    def tearDown(self):
        get_ai_settings.cache_clear()

    def test_default_profile_matches_current_hardcoded_models(self):
        with patch.dict(os.environ, {}, clear=True):
            get_ai_settings.cache_clear()
            settings = get_ai_settings()

        self.assertEqual(settings.profile, "local_current_hardcoded")
        self.assertEqual(settings.decision_llm.model, "qwen3:4b-instruct-2507-q8_0")
        self.assertEqual(settings.response_llm.model, "nollama/mythomax-l2-13b:Q4_K_M")
        self.assertEqual(settings.judge_llm.model, "mistral:7b-instruct-v0.3-q8_0")
        self.assertEqual(settings.embedding_model.model, "mxbai-embed-large")
        self.assertEqual(settings.chroma.path, "./faction_db")
        self.assertEqual(settings.chroma.collection, "factions")

    def test_role_overrides_are_applied(self):
        with patch.dict(os.environ, {
            "AI_PROFILE": "local_current_hardcoded",
            "DECISION_MODEL": "custom-decision",
            "RESPONSE_MODEL": "custom-response",
        }, clear=True):
            get_ai_settings.cache_clear()
            settings = get_ai_settings()

        self.assertEqual(settings.decision_llm.model, "custom-decision")
        self.assertEqual(settings.response_llm.model, "custom-response")

    def test_hosted_provider_requires_base_url(self):
        with patch.dict(os.environ, {
            "AI_PROFILE": "local_current_hardcoded",
            "DECISION_PROVIDER": "huggingface",
            "DECISION_MODEL": "some-model",
        }, clear=True):
            get_ai_settings.cache_clear()
            with self.assertRaises(ValueError):
                get_ai_settings()
