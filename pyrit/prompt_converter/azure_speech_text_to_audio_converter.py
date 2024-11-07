# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
from typing import Literal

import azure.cognitiveservices.speech as speechsdk

from pyrit.common import default_values
from pyrit.models import data_serializer_factory
from pyrit.models import PromptDataType
from pyrit.prompt_converter import ConverterResult, PromptConverter

logger = logging.getLogger(__name__)


class AzureSpeechTextToAudioConverter(PromptConverter):
    """
    The AzureSpeechTextToAudio takes a prompt and generates a wave file.
    https://learn.microsoft.com/en-us/azure/ai-services/speech-service/text-to-speech
    Args:
        azure_speech_region (str): The name of the Azure region.
        azure_speech_key (str): The API key for accessing the service.
        synthesis_language (str): Synthesis voice language
        synthesis_voice_name (str): Synthesis voice name, see URL
        For more details see the following link for synthesis language and synthesis voice:
        https://learn.microsoft.com/en-us/azure/ai-services/speech-service/language-support
        filename (str): File name to be generated.  Please include either .wav or .mp3
        output_format (str): Either wav or mp3. Must match the file prefix.
    """

    AZURE_SPEECH_REGION_ENVIRONMENT_VARIABLE: str = "AZURE_SPEECH_REGION"
    AZURE_SPEECH_KEY_ENVIRONMENT_VARIABLE: str = "AZURE_SPEECH_KEY"

    AzureSpeachAudioFormat = Literal["wav", "mp3"]

    def __init__(
        self,
        azure_speech_region: str = None,
        azure_speech_key: str = None,
        synthesis_language: str = "en_US",
        synthesis_voice_name: str = "en-US-AvaNeural",
        output_format: AzureSpeachAudioFormat = "wav",
    ) -> None:

        self._azure_speech_region: str = default_values.get_required_value(
            env_var_name=self.AZURE_SPEECH_REGION_ENVIRONMENT_VARIABLE, passed_value=azure_speech_region
        )

        self._azure_speech_key: str = default_values.get_required_value(
            env_var_name=self.AZURE_SPEECH_KEY_ENVIRONMENT_VARIABLE, passed_value=azure_speech_key
        )

        self._synthesis_language = synthesis_language
        self._synthesis_voice_name = synthesis_voice_name
        self._output_format = output_format

    def input_supported(self, input_type: PromptDataType) -> bool:
        return input_type == "text"

    async def convert_async(self, *, prompt: str, input_type: PromptDataType = "text") -> ConverterResult:
        if not self.input_supported(input_type):
            raise ValueError("Input type not supported")

        if prompt.strip() == "":
            raise ValueError("Prompt was empty. Please provide valid input prompt.")

        audio_serializer = data_serializer_factory(data_type="audio_path", extension=self._output_format)

        audio_serializer_file = None
        try:
            speech_config = speechsdk.SpeechConfig(
                subscription=self._azure_speech_key,
                region=self._azure_speech_region,
            )
            speech_config.speech_synthesis_language = self._synthesis_language
            speech_config.speech_synthesis_voice_name = self._synthesis_voice_name

            if self._output_format == "mp3":
                speech_config.set_speech_synthesis_output_format(
                    speechsdk.SpeechSynthesisOutputFormat.Audio16Khz32KBitRateMonoMp3
                )

            speech_synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config)

            result = speech_synthesizer.speak_text_async(prompt).get()
            if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
                audio_data = result.audio_data
                await audio_serializer.save_data(audio_data)
                audio_serializer_file = str(audio_serializer.value)
                logger.info(
                    "Speech synthesized for text [{}], and the audio was saved to [{}]".format(
                        prompt, audio_serializer_file
                    )
                )
            elif result.reason == speechsdk.ResultReason.Canceled:
                cancellation_details = result.cancellation_details
                logger.info("Speech synthesis canceled: {}".format(cancellation_details.reason))
                if cancellation_details.reason == speechsdk.CancellationReason.Error:
                    logger.error("Error details: {}".format(cancellation_details.error_details))
                raise RuntimeError(
                    "Speech synthesis canceled: {}".format(cancellation_details.reason)
                    + "Error details: {}".format(cancellation_details.error_details)
                )
        except Exception as e:
            logger.error("Failed to convert prompt to audio: %s", str(e))
            raise
        return ConverterResult(output_text=audio_serializer_file, output_type="audio_path")
