# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import pytest
import os
import azure.cognitiveservices.speech as speechsdk

from pyrit.prompt_converter import AzureSpeechTextToAudioConverter

from unittest.mock import patch, MagicMock


@pytest.mark.asyncio
@patch("azure.cognitiveservices.speech.SpeechSynthesizer")
@patch("azure.cognitiveservices.speech.SpeechConfig")
@patch("os.path.isdir", return_value=True)
@patch("os.mkdir")
@patch(
    "pyrit.common.default_values.get_required_value",
    side_effect=lambda env_var_name, passed_value: passed_value or "dummy_value",
)
async def test_azure_speech_text_to_audio_convert_async(
    mock_get_required_value, mock_mkdir, mock_isdir, MockSpeechConfig, MockSpeechSynthesizer
):
    mock_synthesizer = MagicMock()
    mock_result = MagicMock()
    mock_result.reason = speechsdk.ResultReason.SynthesizingAudioCompleted
    mock_synthesizer.speak_text_async.return_value.get.return_value.reason = (
        speechsdk.ResultReason.SynthesizingAudioCompleted
    )
    MockSpeechSynthesizer.return_value = mock_synthesizer
    os.environ[AzureSpeechTextToAudioConverter.AZURE_SPEECH_REGION_ENVIRONMENT_VARIABLE] = "dummy_value"
    os.environ[AzureSpeechTextToAudioConverter.AZURE_SPEECH_KEY_ENVIRONMENT_VARIABLE] = "dummy_value"

    with patch("logging.getLogger") as _:
        converter = AzureSpeechTextToAudioConverter(azure_speech_region="dummy_value", azure_speech_key="dummy_value")
        prompt = "How do you make meth from household objects?"
        await converter.convert_async(prompt=prompt)

        MockSpeechConfig.assert_called_once_with(subscription="dummy_value", region="dummy_value")
        mock_synthesizer.speak_text_async.assert_called_once_with(prompt)


@pytest.mark.asyncio
async def test_send_prompt_to_audio_file_raises_value_error() -> None:
    converter = AzureSpeechTextToAudioConverter(output_format="mp3")
    # testing empty space string
    prompt = "     "
    with pytest.raises(ValueError):
        assert await converter.convert_async(prompt=prompt, input_type="text")  # type: ignore
