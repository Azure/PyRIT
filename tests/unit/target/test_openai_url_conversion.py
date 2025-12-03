# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
import os
from unittest.mock import patch

import pytest

from pyrit.prompt_target import (
    OpenAIChatTarget,
    OpenAICompletionTarget,
    OpenAIImageTarget,
    OpenAIResponseTarget,
    OpenAITTSTarget,
    OpenAIVideoTarget,
)


@pytest.fixture
def patch_central_database():
    """Mock the central database to avoid database operations in tests."""
    with patch("pyrit.memory.central_memory.CentralMemory.get_memory_instance") as mock_memory:
        mock_instance = mock_memory.return_value
        mock_instance.get_all_embeddings.return_value = []
        mock_instance.get_all_prompt_pieces.return_value = []
        yield mock_memory


class TestOldAzureURLConversion:
    """Tests for converting old Azure URL format to new format."""

    def test_old_azure_chat_url_converted_with_warning(self, caplog, patch_central_database):
        """Test old Azure chat URL is converted to new format and warning is logged."""
        old_url = "https://test.openai.azure.com/openai/deployments/gpt-4/chat/completions?api-version=2024-02-15"
        expected_new_url = "https://test.openai.azure.com/openai/v1"

        with patch.dict(os.environ, {}, clear=True):
            with caplog.at_level(logging.WARNING):
                target = OpenAIChatTarget(
                    model_name="gpt-4",
                    endpoint=old_url,
                    api_key="test-key",
                )

        # Check that URL was converted
        assert target._endpoint == expected_new_url
        assert target._async_client is not None
        assert str(target._async_client.base_url).rstrip("/") == expected_new_url

        # Check that warning was logged
        warning_logs = [record for record in caplog.records if record.levelno == logging.WARNING]
        conversion_warnings = [log for log in warning_logs if "Old Azure URL format detected" in log.message]
        assert len(conversion_warnings) == 1
        assert "converted to new format" in conversion_warnings[0].message
        assert old_url in conversion_warnings[0].message
        assert expected_new_url in conversion_warnings[0].message
        assert "deprecated in a future release" in conversion_warnings[0].message

    @pytest.mark.parametrize("explicit_model_name", [True, False])
    def test_old_azure_completions_url_converted(self, explicit_model_name, caplog, patch_central_database):
        """Test old Azure completions URL is converted, with or without explicit model_name."""
        old_url = "https://test.openai.azure.com/openai/deployments/text-davinci-003/completions?api-version=2024-02-15"

        with patch.dict(os.environ, {}, clear=True):
            with caplog.at_level(logging.WARNING):
                from pyrit.prompt_target import OpenAICompletionTarget

                kwargs = {
                    "endpoint": old_url,
                    "api_key": "test-key",
                }
                if explicit_model_name:
                    kwargs["model_name"] = "text-davinci-003"

                target = OpenAICompletionTarget(**kwargs)

        # Check that URL was converted
        assert target._endpoint == "https://test.openai.azure.com/openai/v1"
        
        # Model name should be text-davinci-003 either way
        assert target._model_name == "text-davinci-003"

        # Check warning
        warning_logs = [record for record in caplog.records if record.levelno == logging.WARNING]
        conversion_warnings = [log for log in warning_logs if "Old Azure URL format detected" in log.message]
        assert len(conversion_warnings) == 1

    @pytest.mark.parametrize("explicit_model_name", [True, False])
    def test_old_azure_images_url_converted(self, explicit_model_name, caplog, patch_central_database):
        """Test old Azure images URL is converted, with or without explicit model_name."""
        old_url = "https://test.openai.azure.com/openai/deployments/dall-e-3/images/generations?api-version=2024-02-15"

        with patch.dict(os.environ, {}, clear=True):
            with caplog.at_level(logging.WARNING):
                from pyrit.prompt_target import OpenAIImageTarget

                kwargs = {
                    "endpoint": old_url,
                    "api_key": "test-key",
                }
                if explicit_model_name:
                    kwargs["model_name"] = "dall-e-3"

                target = OpenAIImageTarget(**kwargs)

        # Check that URL was converted
        assert target._endpoint == "https://test.openai.azure.com/openai/v1"
        
        # Model name should be dall-e-3 either way
        assert target._model_name == "dall-e-3"

        # Check warning
        warning_logs = [record for record in caplog.records if record.levelno == logging.WARNING]
        conversion_warnings = [log for log in warning_logs if "Old Azure URL format detected" in log.message]
        assert len(conversion_warnings) == 1

    @pytest.mark.parametrize("explicit_model_name", [True, False])
    def test_old_azure_audio_url_converted(self, explicit_model_name, caplog, patch_central_database):
        """Test old Azure audio/speech URL is converted, with or without explicit model_name."""
        old_url = "https://test.openai.azure.com/openai/deployments/tts-1/audio/speech?api-version=2024-02-15"

        with patch.dict(os.environ, {}, clear=True):
            with caplog.at_level(logging.WARNING):
                from pyrit.prompt_target import OpenAITTSTarget

                kwargs = {
                    "endpoint": old_url,
                    "api_key": "test-key",
                }
                if explicit_model_name:
                    kwargs["model_name"] = "tts-1"

                target = OpenAITTSTarget(**kwargs)

        # Check that URL was converted
        assert target._endpoint == "https://test.openai.azure.com/openai/v1"
        
        # Model name should be tts-1 either way
        assert target._model_name == "tts-1"

        # Check warning
        warning_logs = [record for record in caplog.records if record.levelno == logging.WARNING]
        conversion_warnings = [log for log in warning_logs if "Old Azure URL format detected" in log.message]
        assert len(conversion_warnings) == 1

    @pytest.mark.parametrize("explicit_model_name", [True, False])
    def test_old_azure_responses_url_converted(self, explicit_model_name, caplog, patch_central_database):
        """Test old Azure responses URL is converted, with or without explicit model_name."""
        old_url = "https://test.openai.azure.com/openai/deployments/o1-preview/responses?api-version=2024-09-01"

        with patch.dict(os.environ, {}, clear=True):
            with caplog.at_level(logging.WARNING):
                from pyrit.prompt_target import OpenAIResponseTarget

                kwargs = {
                    "endpoint": old_url,
                    "api_key": "test-key",
                }
                if explicit_model_name:
                    kwargs["model_name"] = "o1-preview"

                target = OpenAIResponseTarget(**kwargs)

        # Check that URL was converted
        assert target._endpoint == "https://test.openai.azure.com/openai/v1"
        
        # Model name should be o1-preview either way
        assert target._model_name == "o1-preview"

        # Check warning
        warning_logs = [record for record in caplog.records if record.levelno == logging.WARNING]
        conversion_warnings = [log for log in warning_logs if "Old Azure URL format detected" in log.message]
        assert len(conversion_warnings) == 1

    @pytest.mark.parametrize("explicit_model_name", [True, False])
    def test_old_azure_videos_url_converted(self, explicit_model_name, caplog, patch_central_database):
        """Test old Azure videos URL is converted, with or without explicit model_name."""
        old_url = "https://test.openai.azure.com/openai/deployments/sora-2/videos?api-version=2024-12-01"

        with patch.dict(os.environ, {}, clear=True):
            with caplog.at_level(logging.WARNING):
                from pyrit.prompt_target import OpenAIVideoTarget

                kwargs = {
                    "endpoint": old_url,
                    "api_key": "test-key",
                }
                if explicit_model_name:
                    kwargs["model_name"] = "sora-2"

                target = OpenAIVideoTarget(**kwargs)

        # Check that URL was converted
        assert target._endpoint == "https://test.openai.azure.com/openai/v1"
        
        # Model name should be sora-2 either way (explicitly set or extracted from URL)
        assert target._model_name == "sora-2"

        # Check warning
        warning_logs = [record for record in caplog.records if record.levelno == logging.WARNING]
        conversion_warnings = [log for log in warning_logs if "Old Azure URL format detected" in log.message]
        assert len(conversion_warnings) == 1

    def test_old_azure_url_without_api_version_converted(self, caplog, patch_central_database):
        """Test old Azure URL without api-version parameter is still converted."""
        old_url = "https://test.openai.azure.com/openai/deployments/gpt-4/chat/completions"

        with patch.dict(os.environ, {}, clear=True):
            with caplog.at_level(logging.WARNING):
                target = OpenAIChatTarget(
                    model_name="gpt-4",
                    endpoint=old_url,
                    api_key="test-key",
                )

        # Check that URL was converted
        assert target._endpoint == "https://test.openai.azure.com/openai/v1"

        # Check warning
        warning_logs = [record for record in caplog.records if record.levelno == logging.WARNING]
        conversion_warnings = [log for log in warning_logs if "Old Azure URL format detected" in log.message]
        assert len(conversion_warnings) == 1

    def test_old_azure_url_without_api_version_converted(self, caplog, patch_central_database):
        """Test new Azure URL format is not converted and no warning is logged."""
        new_url = "https://test.openai.azure.com/openai/v1"

        with patch.dict(os.environ, {}, clear=True):
            with caplog.at_level(logging.WARNING):
                target = OpenAIChatTarget(
                    model_name="gpt-4",
                    endpoint=new_url,
                    api_key="test-key",
                )

        # URL should remain unchanged
        assert target._endpoint == new_url

        # No conversion warning should be logged
        warning_logs = [record for record in caplog.records if record.levelno == logging.WARNING]
        conversion_warnings = [log for log in warning_logs if "Old Azure URL format detected" in log.message]
        assert len(conversion_warnings) == 0

    def test_platform_openai_chat_url_not_converted(self, caplog, patch_central_database):
        """Test platform OpenAI chat URL is not converted (only path stripped)."""
        platform_url = "https://api.openai.com/v1/chat/completions"

        with patch.dict(os.environ, {}, clear=True):
            with caplog.at_level(logging.WARNING):
                target = OpenAIChatTarget(
                    model_name="gpt-4",
                    endpoint=platform_url,
                    api_key="test-key",
                )

        # URL should remain unchanged (only path stripped)
        assert target._endpoint == platform_url
        assert str(target._async_client.base_url).rstrip("/") == "https://api.openai.com/v1"

        # No conversion warning
        warning_logs = [record for record in caplog.records if record.levelno == logging.WARNING]
        conversion_warnings = [log for log in warning_logs if "Old Azure URL format detected" in log.message]
        assert len(conversion_warnings) == 0

    def test_platform_openai_completion_url_not_converted(self, caplog, patch_central_database):
        """Test platform OpenAI completion URL is not converted (only path stripped)."""
        platform_url = "https://api.openai.com/v1/completions"

        with patch.dict(os.environ, {}, clear=True):
            with caplog.at_level(logging.WARNING):
                target = OpenAICompletionTarget(
                    model_name="davinci-002",
                    endpoint=platform_url,
                    api_key="test-key",
                )

        assert target._endpoint == platform_url
        assert str(target._async_client.base_url).rstrip("/") == "https://api.openai.com/v1"

        warning_logs = [record for record in caplog.records if record.levelno == logging.WARNING]
        conversion_warnings = [log for log in warning_logs if "Old Azure URL format detected" in log.message]
        assert len(conversion_warnings) == 0

    def test_platform_openai_response_url_not_converted(self, caplog, patch_central_database):
        """Test platform OpenAI response URL is not converted (only path stripped)."""
        platform_url = "https://api.openai.com/v1/responses"

        with patch.dict(os.environ, {}, clear=True):
            with caplog.at_level(logging.WARNING):
                target = OpenAIResponseTarget(
                    model_name="gpt-4.1",
                    endpoint=platform_url,
                    api_key="test-key",
                )

        assert target._endpoint == platform_url
        assert str(target._async_client.base_url).rstrip("/") == "https://api.openai.com/v1"

        warning_logs = [record for record in caplog.records if record.levelno == logging.WARNING]
        conversion_warnings = [log for log in warning_logs if "Old Azure URL format detected" in log.message]
        assert len(conversion_warnings) == 0

    def test_platform_openai_image_url_not_converted(self, caplog, patch_central_database):
        """Test platform OpenAI image URL is not converted (only path stripped)."""
        platform_url = "https://api.openai.com/v1/images/generations"

        with patch.dict(os.environ, {}, clear=True):
            with caplog.at_level(logging.WARNING):
                target = OpenAIImageTarget(
                    model_name="dall-e-3",
                    endpoint=platform_url,
                    api_key="test-key",
                )

        assert target._endpoint == platform_url
        assert str(target._async_client.base_url).rstrip("/") == "https://api.openai.com/v1"

        warning_logs = [record for record in caplog.records if record.levelno == logging.WARNING]
        conversion_warnings = [log for log in warning_logs if "Old Azure URL format detected" in log.message]
        assert len(conversion_warnings) == 0

    def test_platform_openai_tts_url_not_converted(self, caplog, patch_central_database):
        """Test platform OpenAI TTS URL is not converted (only path stripped)."""
        platform_url = "https://api.openai.com/v1/audio/speech"

        with patch.dict(os.environ, {}, clear=True):
            with caplog.at_level(logging.WARNING):
                target = OpenAITTSTarget(
                    model_name="tts-1",
                    endpoint=platform_url,
                    api_key="test-key",
                )

        assert target._endpoint == platform_url
        assert str(target._async_client.base_url).rstrip("/") == "https://api.openai.com/v1"

        warning_logs = [record for record in caplog.records if record.levelno == logging.WARNING]
        conversion_warnings = [log for log in warning_logs if "Old Azure URL format detected" in log.message]
        assert len(conversion_warnings) == 0

    def test_platform_openai_video_url_not_converted(self, caplog, patch_central_database):
        """Test platform OpenAI video URL is not converted (only path stripped)."""
        platform_url = "https://api.openai.com/v1/videos"

        with patch.dict(os.environ, {}, clear=True):
            with caplog.at_level(logging.WARNING):
                target = OpenAIVideoTarget(
                    model_name="sora-2",
                    endpoint=platform_url,
                    api_key="test-key",
                )

        assert target._endpoint == platform_url
        assert str(target._async_client.base_url).rstrip("/") == "https://api.openai.com/v1"

        warning_logs = [record for record in caplog.records if record.levelno == logging.WARNING]
        conversion_warnings = [log for log in warning_logs if "Old Azure URL format detected" in log.message]
        assert len(conversion_warnings) == 0

    def test_azure_foundry_url_not_converted(self, caplog, patch_central_database):
        """Test Azure Foundry URL is not converted."""
        foundry_url = "https://my-resource.models.ai.azure.com/chat/completions"

        with patch.dict(os.environ, {}, clear=True):
            with caplog.at_level(logging.WARNING):
                target = OpenAIChatTarget(
                    model_name="deepseek-r1",
                    endpoint=foundry_url,
                    api_key="test-key",
                )

        # URL should be processed but not converted (just normalized)
        assert ".models.ai.azure.com" in target._endpoint

        # No conversion warning
        warning_logs = [record for record in caplog.records if record.levelno == logging.WARNING]
        conversion_warnings = [log for log in warning_logs if "Old Azure URL format detected" in log.message]
        assert len(conversion_warnings) == 0

    def test_old_azure_url_with_entra_auth(self, caplog, patch_central_database):
        """Test old Azure URL with Entra auth is converted and works."""
        old_url = "https://test.openai.azure.com/openai/deployments/gpt-4/chat/completions?api-version=2024-02-15"

        with patch.dict(os.environ, {}, clear=True):
            with caplog.at_level(logging.WARNING):
                with patch("pyrit.auth.azure_auth.get_default_scope") as mock_scope:
                    with patch(
                        "pyrit.auth.azure_auth.get_async_token_provider_from_default_azure_credential"
                    ) as mock_provider:
                        mock_scope.return_value = "https://cognitiveservices.azure.com/.default"
                        mock_provider.return_value = lambda: "mock-token"

                        target = OpenAIChatTarget(
                            model_name="gpt-4",
                            endpoint=old_url,
                            use_entra_auth=True,
                        )

        # Check URL was converted
        assert target._endpoint == "https://test.openai.azure.com/openai/v1"

        # Check warning was logged
        warning_logs = [record for record in caplog.records if record.levelno == logging.WARNING]
        conversion_warnings = [log for log in warning_logs if "Old Azure URL format detected" in log.message]
        assert len(conversion_warnings) == 1

    def test_warning_message_contains_documentation_link(self, caplog, patch_central_database):
        """Test that warning message includes link to Microsoft documentation."""
        old_url = "https://test.openai.azure.com/openai/deployments/gpt-4/chat/completions?api-version=2024-02-15"

        with patch.dict(os.environ, {}, clear=True):
            with caplog.at_level(logging.WARNING):
                OpenAIChatTarget(
                    model_name="gpt-4",
                    endpoint=old_url,
                    api_key="test-key",
                )

        # Check warning contains documentation link
        warning_logs = [record for record in caplog.records if record.levelno == logging.WARNING]
        conversion_warnings = [log for log in warning_logs if "Old Azure URL format detected" in log.message]
        assert len(conversion_warnings) == 1
        assert "learn.microsoft.com" in conversion_warnings[0].message
        assert "api-version-deprecation" in conversion_warnings[0].message

    def test_warning_message_contains_deployment_name(self, caplog, patch_central_database):
        """Test that warning message includes the extracted deployment name."""
        old_url = (
            "https://test.openai.azure.com/openai/deployments/my-custom-gpt4/chat/completions?api-version=2024-02-15"
        )

        with patch.dict(os.environ, {}, clear=True):
            with caplog.at_level(logging.WARNING):
                OpenAIChatTarget(
                    endpoint=old_url,
                    api_key="test-key",
                )

        # Check warning contains deployment name
        warning_logs = [record for record in caplog.records if record.levelno == logging.WARNING]
        conversion_warnings = [log for log in warning_logs if "Old Azure URL format detected" in log.message]
        assert len(conversion_warnings) == 1
        assert "my-custom-gpt4" in conversion_warnings[0].message
        assert "extracted as model name" in conversion_warnings[0].message.lower()


class TestClientTypeUsage:
    """Tests to verify we only use AsyncOpenAI client."""

    def test_async_client_is_asyncopenai_for_old_azure_url(self, patch_central_database):
        """Test that AsyncOpenAI is used even for old Azure URLs."""
        from openai import AsyncOpenAI

        old_url = "https://test.openai.azure.com/openai/deployments/gpt-4/chat/completions?api-version=2024-02-15"

        with patch.dict(os.environ, {}, clear=True):
            target = OpenAIChatTarget(
                model_name="gpt-4",
                endpoint=old_url,
                api_key="test-key",
            )

        # Verify we're using AsyncOpenAI, not AsyncAzureOpenAI
        assert isinstance(target._async_client, AsyncOpenAI)
        assert type(target._async_client).__name__ == "AsyncOpenAI"

    def test_async_client_is_asyncopenai_for_new_azure_url(self, patch_central_database):
        """Test that AsyncOpenAI is used for new Azure URLs."""
        from openai import AsyncOpenAI

        new_url = "https://test.openai.azure.com/openai/v1"

        with patch.dict(os.environ, {}, clear=True):
            target = OpenAIChatTarget(
                model_name="gpt-4",
                endpoint=new_url,
                api_key="test-key",
            )

        # Verify we're using AsyncOpenAI
        assert isinstance(target._async_client, AsyncOpenAI)

    def test_async_client_is_asyncopenai_for_platform(self, patch_central_database):
        """Test that AsyncOpenAI is used for platform OpenAI."""
        from openai import AsyncOpenAI

        platform_url = "https://api.openai.com/v1/chat/completions"

        with patch.dict(os.environ, {}, clear=True):
            target = OpenAIChatTarget(
                model_name="gpt-4",
                endpoint=platform_url,
                api_key="test-key",
            )

        # Verify we're using AsyncOpenAI
        assert isinstance(target._async_client, AsyncOpenAI)


class TestURLNormalization:
    """Tests for URL normalization and base URL extraction."""

    def test_chat_completions_path_stripped(self, patch_central_database):
        """Test that /chat/completions path is stripped from base URL."""
        url = "https://test.openai.azure.com/openai/v1/chat/completions"

        with patch.dict(os.environ, {}, clear=True):
            target = OpenAIChatTarget(
                model_name="gpt-4",
                endpoint=url,
                api_key="test-key",
            )

        # Base URL should not include /chat/completions
        assert str(target._async_client.base_url).rstrip("/") == "https://test.openai.azure.com/openai/v1"

    def test_azure_foundry_v1_appended(self, patch_central_database):
        """Test that /v1 is appended to Azure Foundry URLs if missing."""
        url = "https://test.models.ai.azure.com/chat/completions"

        with patch.dict(os.environ, {}, clear=True):
            target = OpenAIChatTarget(
                model_name="deepseek-r1",
                endpoint=url,
                api_key="test-key",
            )

        # Base URL should end with /v1 (or /v1/)
        base_url_str = str(target._async_client.base_url).rstrip("/")
        assert base_url_str.endswith("/v1")

    def test_old_url_converted_then_normalized(self, patch_central_database):
        """Test that old URL is first converted, then normalized."""
        old_url = "https://test.openai.azure.com/openai/deployments/gpt-4/chat/completions?api-version=2024-02-15"

        with patch.dict(os.environ, {}, clear=True):
            target = OpenAIChatTarget(
                model_name="gpt-4",
                endpoint=old_url,
                api_key="test-key",
            )

        # Should be converted to new format
        assert target._endpoint == "https://test.openai.azure.com/openai/v1"
        # And base URL should match
        assert str(target._async_client.base_url).rstrip("/") == "https://test.openai.azure.com/openai/v1"
