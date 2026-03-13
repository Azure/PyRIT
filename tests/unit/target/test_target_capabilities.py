# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import pytest

from pyrit.prompt_target.common.target_capabilities import TargetCapabilities


@pytest.mark.usefixtures("patch_central_database")
class TestTargetCapabilitiesModalities:
    """Test that each target declares the correct input/output modalities via _DEFAULT_CAPABILITIES."""

    def test_default_capabilities_are_text_only(self):
        caps = TargetCapabilities()
        assert caps.input_modalities == ["text"]
        assert caps.output_modalities == ["text"]

    def test_openai_chat_target_modalities(self):
        from pyrit.prompt_target import OpenAIChatTarget

        target = OpenAIChatTarget(
            model_name="test-model",
            endpoint="https://mock.azure.com/",
            api_key="mock-api-key",
        )
        assert "text" in target.capabilities.input_modalities
        assert "image_path" in target.capabilities.input_modalities
        assert "audio_path" in target.capabilities.input_modalities
        assert "text" in target.capabilities.output_modalities
        assert "audio_path" in target.capabilities.output_modalities

    def test_openai_image_target_modalities(self):
        from pyrit.prompt_target import OpenAIImageTarget

        target = OpenAIImageTarget(
            model_name="dall-e-3",
            endpoint="https://mock.azure.com/",
            api_key="mock-api-key",
        )
        assert "text" in target.capabilities.input_modalities
        assert "image_path" in target.capabilities.input_modalities
        assert target.capabilities.output_modalities == ["image_path"]

    def test_openai_tts_target_modalities(self):
        from pyrit.prompt_target import OpenAITTSTarget

        target = OpenAITTSTarget(
            model_name="tts-1",
            endpoint="https://mock.azure.com/",
            api_key="mock-api-key",
        )
        assert target.capabilities.input_modalities == ["text"]
        assert target.capabilities.output_modalities == ["audio_path"]

    def test_openai_video_target_modalities(self):
        from pyrit.prompt_target import OpenAIVideoTarget

        target = OpenAIVideoTarget(
            model_name="sora-2",
            endpoint="https://mock.azure.com/",
            api_key="mock-api-key",
        )
        assert "text" in target.capabilities.input_modalities
        assert "image_path" in target.capabilities.input_modalities
        assert target.capabilities.output_modalities == ["video_path"]

    def test_openai_realtime_target_modalities(self):
        from pyrit.prompt_target import RealtimeTarget

        target = RealtimeTarget(
            model_name="gpt-4o-realtime",
            endpoint="https://mock.azure.com/",
            api_key="mock-api-key",
        )
        assert "text" in target.capabilities.input_modalities
        assert "audio_path" in target.capabilities.input_modalities
        assert "text" in target.capabilities.output_modalities
        assert "audio_path" in target.capabilities.output_modalities

    def test_openai_response_target_modalities(self):
        from pyrit.prompt_target import OpenAIResponseTarget

        target = OpenAIResponseTarget(
            model_name="o1",
            endpoint="https://mock.azure.com/",
            api_key="mock-api-key",
        )
        assert "text" in target.capabilities.input_modalities
        assert "image_path" in target.capabilities.input_modalities
        assert target.capabilities.output_modalities == ["text"]

    def test_openai_completion_target_modalities(self):
        from pyrit.prompt_target import OpenAICompletionTarget

        target = OpenAICompletionTarget(
            model_name="test-model",
            endpoint="https://mock.azure.com/",
            api_key="mock-api-key",
        )
        assert target.capabilities.input_modalities == ["text"]
        assert target.capabilities.output_modalities == ["text"]

    def test_azure_blob_storage_target_modalities(self):
        from pyrit.prompt_target import AzureBlobStorageTarget

        target = AzureBlobStorageTarget(
            container_url="https://mock.blob.core.windows.net/container",
            sas_token="mock-sas-token",
        )
        assert "text" in target.capabilities.input_modalities
        assert "url" in target.capabilities.input_modalities
        assert target.capabilities.output_modalities == ["url"]

    def test_text_target_modalities(self):
        from pyrit.prompt_target import TextTarget

        target = TextTarget()
        assert target.capabilities.input_modalities == ["text"]
        assert target.capabilities.output_modalities == ["text"]

    def test_playwright_target_modalities(self):
        from unittest.mock import MagicMock

        from pyrit.prompt_target import PlaywrightTarget

        target = PlaywrightTarget(
            interaction_func=MagicMock(),
            page=MagicMock(),
        )
        assert "text" in target.capabilities.input_modalities
        assert "image_path" in target.capabilities.input_modalities
        assert target.capabilities.output_modalities == ["text"]

    def test_playwright_copilot_target_modalities(self):
        from unittest.mock import MagicMock

        from pyrit.prompt_target import PlaywrightCopilotTarget

        target = PlaywrightCopilotTarget(page=MagicMock())
        assert "text" in target.capabilities.input_modalities
        assert "image_path" in target.capabilities.input_modalities
        assert "text" in target.capabilities.output_modalities
        assert "image_path" in target.capabilities.output_modalities

    def test_websocket_copilot_target_modalities(self):
        from unittest.mock import MagicMock

        from pyrit.prompt_target import WebSocketCopilotTarget

        target = WebSocketCopilotTarget(authenticator=MagicMock())
        assert "text" in target.capabilities.input_modalities
        assert "image_path" in target.capabilities.input_modalities
        assert target.capabilities.output_modalities == ["text"]

    def test_custom_capabilities_override_modalities(self):
        from pyrit.prompt_target import OpenAIChatTarget, TargetCapabilities

        custom = TargetCapabilities(
            supports_multi_turn=True,
            input_modalities=["text"],
            output_modalities=["text"],
        )
        target = OpenAIChatTarget(
            model_name="test-model",
            endpoint="https://mock.azure.com/",
            api_key="mock-api-key",
            custom_capabilities=custom,
        )
        assert target.capabilities.input_modalities == ["text"]
        assert target.capabilities.output_modalities == ["text"]


class TestTargetCapabilitiesAssertSatisfies:
    """Test the assert_satifies method including list-based modality checks."""

    def test_assert_satisfies_passes_when_all_met(self):
        caps = TargetCapabilities(
            supports_multi_turn=True,
            input_modalities=["text", "image_path"],
            output_modalities=["text"],
        )
        required = TargetCapabilities(
            supports_multi_turn=True,
            input_modalities=["text"],
            output_modalities=["text"],
        )
        caps.assert_satifies(required)  # should not raise

    def test_assert_satisfies_fails_on_unmet_bool(self):
        caps = TargetCapabilities(supports_multi_turn=False)
        required = TargetCapabilities(supports_multi_turn=True)
        with pytest.raises(ValueError, match="supports_multi_turn"):
            caps.assert_satifies(required)

    def test_assert_satisfies_fails_on_missing_input_modality(self):
        caps = TargetCapabilities(input_modalities=["text"])
        required = TargetCapabilities(input_modalities=["text", "image_path"])
        with pytest.raises(ValueError, match="input_modalities"):
            caps.assert_satifies(required)

    def test_assert_satisfies_fails_on_missing_output_modality(self):
        caps = TargetCapabilities(output_modalities=["text"])
        required = TargetCapabilities(output_modalities=["text", "audio_path"])
        with pytest.raises(ValueError, match="output_modalities"):
            caps.assert_satifies(required)

    def test_assert_satisfies_passes_when_superset_of_required_modalities(self):
        caps = TargetCapabilities(
            input_modalities=["text", "image_path", "audio_path"],
            output_modalities=["text", "audio_path"],
        )
        required = TargetCapabilities(
            input_modalities=["text", "image_path"],
            output_modalities=["text"],
        )
        caps.assert_satifies(required)  # should not raise

    def test_assert_satisfies_passes_with_default_text_modalities(self):
        caps = TargetCapabilities()
        required = TargetCapabilities()
        caps.assert_satifies(required)  # should not raise

    def test_assert_satisfies_fails_multiple_unmet(self):
        caps = TargetCapabilities(
            supports_multi_turn=False,
            input_modalities=["text"],
        )
        required = TargetCapabilities(
            supports_multi_turn=True,
            input_modalities=["text", "image_path"],
        )
        with pytest.raises(ValueError, match="supports_multi_turn") as exc_info:
            caps.assert_satifies(required)
        assert "input_modalities" in str(exc_info.value)

    def test_assert_satisfies_ignores_false_required_bools(self):
        """When a required capability bool is False, it should not be flagged as unmet."""
        caps = TargetCapabilities(supports_multi_turn=False)
        required = TargetCapabilities(supports_multi_turn=False)
        caps.assert_satifies(required)  # should not raise
