# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import logging
import pathlib
import os
import uuid

import azure.cognitiveservices.speech as speechsdk
from pyrit.memory.memory_models import PromptDataType
from pyrit.prompt_converter import PromptConverter
from pyrit.common import default_values
from pyrit.common.path import RESULTS_PATH

logger = logging.getLogger(__name__)


class TextToAudioConverter(PromptConverter):
    """
    The TextToAudio takes a prompt and generates a
    wave file.

    Args:
        azure_speech_region (str): The name of the Azure region.
        azure_speech_key (str): The API key for accessing the service.
        synthesis_language (str): Synthesis voice language
        synthesis_voice_name (str): Synthesis voice name, see URL
        https://learn.microsoft.com/en-us/azure/ai-services/speech-service/language-support
        filename (str): File name to be generated.  Please include either .wav or .mp3
        output_format (str): Either wav or mp3. Must match the file prefix.
    """

    AZURE_SPEECH_REGION_ENVIRONMENT_VARIABLE: str = "AZURE_SPEECH_REGION"
    AZURE_SPEECH_KEY_TOKEN_ENVIRONMENT_VARIABLE: str = "AZURE_SPEECH_KEY_TOKEN"
    SUPPORTED_OUTPUT_FORMATS = ["wav", "mp3"]

    def __init__(
        self,
        filename: str = None,
        azure_speech_region: str = None,
        azure_speech_key: str = None,
        synthesis_language: str = "en_US",
        synthesis_voice_name: str = "en-US-AvaNeural",
        output_format: str = "wav",
    ) -> None:

        self.filename = filename
        if output_format not in self.SUPPORTED_OUTPUT_FORMATS:
            raise ValueError(
                f"Invalid output format {output_format}. Supported output formats are {self.SUPPORTED_OUTPUT_FORMATS}"
            )

        self.azure_speech_region: str = default_values.get_required_value(
            env_var_name=self.AZURE_SPEECH_REGION_ENVIRONMENT_VARIABLE, passed_value=azure_speech_region
        )

        self.azure_speech_key: str = default_values.get_required_value(
            env_var_name=self.AZURE_SPEECH_KEY_TOKEN_ENVIRONMENT_VARIABLE, passed_value=azure_speech_key
        )

        self.synthesis_language = synthesis_language

        self.synthesis_voice_name = synthesis_voice_name

        self.output_dir = pathlib.Path(RESULTS_PATH) / "audio"

        self.output_format = output_format

    def is_supported(self, input_type: PromptDataType) -> bool:
        return input_type == "text"

    def send_prompt_to_audio_file(self, prompt: str, output_format: str):
        """
        Takes a prompt and it creates either an MP3 or WAV file.
        Saves the file to the results/audio folder

        Raises:
            ValueError: Any issues in validation or execution.
        """
        if prompt == "":
            raise ValueError("Prompt was empty. Please provide valid input prompt.")
        try:
            speech_config = speechsdk.SpeechConfig(subscription=self.azure_speech_key, region=self.azure_speech_region)
            speech_config.speech_synthesis_language = self.synthesis_language
            speech_config.speech_synthesis_voice_name = self.synthesis_voice_name
            if output_format == "mp3":
                speech_config.set_speech_synthesis_output_format(
                    speechsdk.SpeechSynthesisOutputFormat.Audio16Khz32KBitRateMonoMp3
                )
            if not self.filename:
                self.filename = f"{uuid.uuid4()}.wav"
            if not os.path.isdir(self.output_dir):
                os.mkdir(self.output_dir)
            file_name = os.path.join(self.output_dir, self.filename)
            file_config = speechsdk.audio.AudioOutputConfig(filename=file_name)
            speech_synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=file_config)
            result = speech_synthesizer.speak_text_async(prompt).get()
            if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
                logger.info(
                    "Speech synthesized for text [{}], and the audio was saved to [{}]".format(prompt, file_name)
                )
            elif result.reason == speechsdk.ResultReason.Canceled:
                cancellation_details = result.cancellation_details
                logger.info("Speech synthesis canceled: {}".format(cancellation_details.reason))
                if cancellation_details.reason == speechsdk.CancellationReason.Error:
                    logger.error("Error details: {}".format(cancellation_details.error_details))
        except Exception as e:
            logger.error("Failed to convert prompt to audio: %s", str(e))
            raise

    def convert(self, *, prompt: str, input_type: PromptDataType = "text") -> None:
        if not self.is_supported(input_type):
            raise ValueError("Input type not supported")
        self.send_prompt_to_audio_file(prompt, self.output_format)
