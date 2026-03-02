# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import pytest

from tests.unit.mocks import MockPromptTarget


@pytest.mark.usefixtures("patch_central_database")
class TestSupportsMultiTurn:
    """Test the supports_multi_turn property across the target hierarchy."""

    def test_prompt_target_defaults_to_false(self, patch_central_database):
        # PromptTarget is abstract, so we use a concrete subclass
        # MockPromptTarget inherits from PromptChatTarget, so use a PromptTarget mock
        from unittest.mock import MagicMock

        from pyrit.prompt_target import PromptTarget

        target = MagicMock(spec=PromptTarget)
        # The property should be False on PromptTarget
        assert PromptTarget.supports_multi_turn.fget(target) is False

    def test_prompt_chat_target_returns_true(self, patch_central_database):
        target = MockPromptTarget()
        assert target.supports_multi_turn is True

    def test_openai_chat_target_returns_true(self, patch_central_database):
        from pyrit.prompt_target import OpenAIChatTarget

        target = OpenAIChatTarget(
            model_name="test-model",
            endpoint="https://mock.azure.com/",
            api_key="mock-api-key",
        )
        assert target.supports_multi_turn is True

    def test_openai_image_target_returns_false(self, patch_central_database):
        from pyrit.prompt_target import OpenAIImageTarget

        target = OpenAIImageTarget(
            model_name="dall-e-3",
            endpoint="https://mock.azure.com/",
            api_key="mock-api-key",
        )
        assert target.supports_multi_turn is False

    def test_openai_video_target_returns_false(self, patch_central_database):
        from pyrit.prompt_target import OpenAIVideoTarget

        target = OpenAIVideoTarget(
            model_name="sora-2",
            endpoint="https://mock.azure.com/",
            api_key="mock-api-key",
        )
        assert target.supports_multi_turn is False

    def test_openai_tts_target_returns_false(self, patch_central_database):
        from pyrit.prompt_target import OpenAITTSTarget

        target = OpenAITTSTarget(
            model_name="tts-1",
            endpoint="https://mock.azure.com/",
            api_key="mock-api-key",
        )
        assert target.supports_multi_turn is False

    def test_openai_completion_target_returns_false(self, patch_central_database):
        from pyrit.prompt_target import OpenAICompletionTarget

        target = OpenAICompletionTarget(
            model_name="test-model",
            endpoint="https://mock.azure.com/",
            api_key="mock-api-key",
        )
        assert target.supports_multi_turn is False

    def test_text_target_returns_false(self, patch_central_database):
        from pyrit.prompt_target import TextTarget

        target = TextTarget()
        assert target.supports_multi_turn is False

    def test_prompt_shield_target_returns_false(self, patch_central_database):
        from pyrit.prompt_target import PromptShieldTarget

        target = PromptShieldTarget(
            endpoint="https://mock.azure.com/",
            api_key="mock-api-key",
        )
        assert target.supports_multi_turn is False
