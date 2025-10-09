# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from unittest.mock import MagicMock, patch

import pytest

from pyrit.prompt_converter import AzureSpeechAudioToTextConverter


def is_speechsdk_installed():
    try:
        import azure.cognitiveservices.speech  # noqa: F401

        return True
    except ModuleNotFoundError:
        return False


@pytest.mark.skipif(not is_speechsdk_installed(), reason="Azure Speech SDK is not installed.")
class TestAzureSpeechAudioToTextConverter:

    @patch(
        "pyrit.common.default_values.get_required_value", side_effect=lambda env_var_name, passed_value: passed_value
    )
    def test_azure_speech_audio_text_converter_initialization(self, mock_get_required_value):
        converter = AzureSpeechAudioToTextConverter(
            azure_speech_region="dummy_region", azure_speech_key="dummy_key", recognition_language="es-ES"
        )
        assert converter._recognition_language == "es-ES"
        assert converter._azure_speech_region == "dummy_region"
        assert converter._azure_speech_key == "dummy_key"
        assert converter.done is False

    @patch("azure.cognitiveservices.speech.SpeechRecognizer")
    @patch("pyrit.prompt_converter.azure_speech_audio_to_text_converter.logger")
    def test_stop_cb(self, mock_logger, MockSpeechRecognizer):
        import azure.cognitiveservices.speech as speechsdk

        # Create a mock event
        mock_event = MagicMock()
        mock_event.result.reason = speechsdk.ResultReason.Canceled
        mock_event.result.cancellation_details.reason = speechsdk.CancellationReason.EndOfStream
        mock_event.result.cancellation_details.error_details = "Mock error details"

        MockSpeechRecognizer.return_value = MagicMock()

        mock_logger.return_value = MagicMock()

        # Call the stop_cb function with the mock event
        converter = AzureSpeechAudioToTextConverter()
        converter.stop_cb(evt=mock_event, recognizer=MockSpeechRecognizer)

        # Check if the callback function worked as expected
        MockSpeechRecognizer.stop_continuous_recognition_async.assert_called_once()
        mock_logger.info.assert_any_call("CLOSING on {}".format(mock_event))
        mock_logger.info.assert_any_call(
            "Speech recognition canceled: {}".format(speechsdk.CancellationReason.EndOfStream)
        )
        mock_logger.info.assert_called_with("End of audio stream detected.")

    @patch("azure.cognitiveservices.speech.SpeechRecognizer")
    @patch("pyrit.prompt_converter.azure_speech_audio_to_text_converter.logger")
    def test_transcript_cb(self, mock_logger, MockSpeechRecognizer):
        import azure.cognitiveservices.speech as speechsdk

        # Create a mock event
        mock_event = MagicMock()
        mock_event.result.reason = speechsdk.ResultReason.RecognizedSpeech
        mock_event.result.text = "Mock transcribed text"

        MockSpeechRecognizer.return_value = MagicMock()

        mock_logger.return_value = MagicMock()

        # Call the transcript_cb function with the mock event
        converter = AzureSpeechAudioToTextConverter()
        transcript = ["sample", "transcript"]
        converter.transcript_cb(evt=mock_event, transcript=transcript)

        # Check if the callback function worked as expected
        mock_logger.info.assert_called_once_with("RECOGNIZED: {}".format(mock_event.result.text))
        assert mock_event.result.text in transcript

    def test_azure_speech_audio_text_converter_input_supported(self):
        converter = AzureSpeechAudioToTextConverter()
        assert converter.input_supported("image_path") is False
        assert converter.input_supported("audio_path") is True

    @pytest.mark.asyncio
    async def test_azure_speech_audio_text_converter_nonexistent_path(self):
        converter = AzureSpeechAudioToTextConverter()
        prompt = "nonexistent_path2.wav"
        with pytest.raises(FileNotFoundError):
            assert await converter.convert_async(prompt=prompt, input_type="audio_path")

    @pytest.mark.asyncio
    @patch("os.path.exists", return_value=True)
    async def test_azure_speech_audio_text_converter_non_wav_file(self, mock_path_exists):
        converter = AzureSpeechAudioToTextConverter()
        prompt = "dummy_audio.mp3"
        with pytest.raises(ValueError):
            assert await converter.convert_async(prompt=prompt, input_type="audio_path")

    def test_use_entra_auth_true_with_api_key_raises_error(self):
        """Test that use_entra_auth=True with api_key raises ValueError."""
        with pytest.raises(ValueError, match="If using Entra ID auth, please do not specify azure_speech_key"):
            AzureSpeechAudioToTextConverter(
                azure_speech_region="test_region", azure_speech_key="test_key", use_entra_auth=True
            )

    @patch(
        "pyrit.prompt_converter.azure_speech_audio_to_text_converter." "get_speech_config_from_default_azure_credential"
    )
    @patch("pyrit.common.default_values.get_required_value")
    def test_use_entra_auth_true_uses_credential(self, mock_get_required_value, mock_get_speech_config):
        """Test that use_entra_auth=True uses get_speech_config_from_default_azure_credential."""
        mock_get_required_value.side_effect = lambda env_var_name, passed_value: passed_value or "mock_value"
        mock_speech_config = MagicMock()
        mock_get_speech_config.return_value = mock_speech_config

        converter = AzureSpeechAudioToTextConverter(
            azure_speech_region="test_region", azure_speech_resource_id="test_resource_id", use_entra_auth=True
        )

        # Call _get_speech_config to trigger the mocked function
        speech_config = converter._get_speech_config()

        # Verify get_speech_config_from_default_azure_credential was called
        mock_get_speech_config.assert_called_once_with(resource_id="test_resource_id", region="test_region")
        assert speech_config == mock_speech_config

    @patch("azure.cognitiveservices.speech.SpeechConfig")
    @patch("pyrit.common.default_values.get_required_value")
    def test_use_entra_auth_false_uses_api_key(self, mock_get_required_value, mock_speech_config_class):
        """Test that use_entra_auth=False uses SpeechConfig with API key."""
        mock_get_required_value.side_effect = lambda env_var_name, passed_value: passed_value or "mock_value"
        mock_speech_config = MagicMock()
        mock_speech_config_class.return_value = mock_speech_config

        converter = AzureSpeechAudioToTextConverter(
            azure_speech_region="test_region", azure_speech_key="test_key", use_entra_auth=False
        )

        # Call _get_speech_config to trigger the mocked function
        speech_config = converter._get_speech_config()

        # Verify SpeechConfig was called with API key
        mock_speech_config_class.assert_called_once_with(subscription="test_key", region="test_region")
        assert speech_config == mock_speech_config

    @patch("azure.cognitiveservices.speech.SpeechConfig")
    @patch("pyrit.common.default_values.get_required_value")
    def test_use_entra_auth_default_false_uses_api_key(self, mock_get_required_value, mock_speech_config_class):
        """Test that default behavior (use_entra_auth=False) uses API key authentication."""
        mock_get_required_value.side_effect = lambda env_var_name, passed_value: passed_value or "mock_value"
        mock_speech_config = MagicMock()
        mock_speech_config_class.return_value = mock_speech_config

        converter = AzureSpeechAudioToTextConverter(
            azure_speech_region="test_region",
            azure_speech_key="test_key",
            # use_entra_auth not specified, should default to False
        )

        # Call _get_speech_config to trigger the mocked function
        speech_config = converter._get_speech_config()

        # Verify SpeechConfig was called with API key (not Entra auth)
        mock_speech_config_class.assert_called_once_with(subscription="test_key", region="test_region")
        assert speech_config == mock_speech_config
