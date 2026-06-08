import unittest
from unittest.mock import MagicMock, patch

from ai.providers import (
    HuggingFaceChatProvider,
    HuggingFaceEmbeddingProvider,
    HuggingFaceTextGenerationProvider,
    NormalizedToolCall,
    RoleProviderConfig,
    create_chat_provider,
    create_embedding_provider,
    create_text_generation_provider,
)


class HuggingFaceProviderTests(unittest.TestCase):
    def setUp(self):
        self.config = RoleProviderConfig(
            provider="huggingface",
            model="test-model",
            hf_provider="featherless-ai",
            api_key_env="HF_TOKEN",
            timeout_seconds=42,
        )

    @patch("ai.providers.InferenceClient")
    def test_factory_returns_huggingface_specific_providers(self, inference_client_cls):
        chat_provider = create_chat_provider(self.config)
        embedding_provider = create_embedding_provider(self.config)
        text_provider = create_text_generation_provider(self.config)

        self.assertIsInstance(chat_provider, HuggingFaceChatProvider)
        self.assertIsInstance(embedding_provider, HuggingFaceEmbeddingProvider)
        self.assertIsInstance(text_provider, HuggingFaceTextGenerationProvider)
        self.assertEqual(inference_client_cls.call_count, 3)

    @patch("ai.providers.InferenceClient")
    def test_chat_provider_uses_text_generation_for_featherless(self, inference_client_cls):
        client = MagicMock()
        inference_client_cls.return_value = client
        client.text_generation.return_value = "hello from featherless"

        provider = HuggingFaceChatProvider(self.config)
        response = provider.chat(messages=[
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "hi"},
        ])

        self.assertEqual(response.content, "hello from featherless")
        self.assertEqual(response.tool_calls, [])
        client.text_generation.assert_called_once_with(
            prompt="system: You are helpful.\n\nuser: hi",
            model="test-model",
        )
        client.chat_completion.assert_not_called()

    @patch("ai.providers.InferenceClient")
    def test_chat_provider_normalizes_inference_client_output(self, inference_client_cls):
        client = MagicMock()
        inference_client_cls.return_value = client
        config = RoleProviderConfig(
            provider="huggingface",
            model="test-model",
            hf_provider="hf-inference",
            api_key_env="HF_TOKEN",
            timeout_seconds=42,
        )

        message = MagicMock()
        message.content = "hello"
        message.tool_calls = [
            {
                "function": {
                    "name": "flag_jailbreak",
                    "arguments": "{\"normalized_user_prompt\": \"safe\"}",
                }
            }
        ]
        choice = MagicMock()
        choice.message = message
        result = MagicMock()
        result.choices = [choice]
        client.chat_completion.return_value = result

        provider = HuggingFaceChatProvider(config)
        response = provider.chat(messages=[{"role": "user", "content": "hi"}])

        self.assertEqual(response.content, "hello")
        self.assertEqual(len(response.tool_calls), 1)
        self.assertIsInstance(response.tool_calls[0], NormalizedToolCall)
        self.assertEqual(response.tool_calls[0].function.name, "flag_jailbreak")
        self.assertEqual(
            response.tool_calls[0].function.arguments,
            {"normalized_user_prompt": "safe"},
        )
        inference_client_cls.assert_called_with(
            timeout=42.0,
            provider="hf-inference",
        )

    @patch("ai.providers.InferenceClient")
    def test_embedding_provider_wraps_single_embedding_row(self, inference_client_cls):
        client = MagicMock()
        inference_client_cls.return_value = client
        client.feature_extraction.return_value = [0.1, 0.2, 0.3]

        provider = HuggingFaceEmbeddingProvider(self.config)

        self.assertEqual(provider.embed("hello"), [[0.1, 0.2, 0.3]])

    @patch("ai.providers.InferenceClient")
    def test_text_generation_provider_uses_inference_client_api(self, inference_client_cls):
        client = MagicMock()
        inference_client_cls.return_value = client
        client.text_generation.return_value = "generated"

        provider = HuggingFaceTextGenerationProvider(self.config)

        self.assertEqual(provider.generate("prompt"), "generated")

    @patch("ai.providers.InferenceClient")
    def test_hf_provider_is_optional(self, inference_client_cls):
        config = RoleProviderConfig(
            provider="huggingface",
            model="test-model",
            api_key_env="HF_TOKEN",
            timeout_seconds=42,
        )

        HuggingFaceChatProvider(config)

        inference_client_cls.assert_called_with(timeout=42.0)


if __name__ == "__main__":
    unittest.main()
