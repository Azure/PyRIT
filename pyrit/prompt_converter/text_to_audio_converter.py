# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import logging
import pathlib

# !pip install azure-cognitiveservices-speech
import azure.cognitiveservices.speech as speechsdk

from pyrit.common.path import RESULTS_PATH
from pyrit.prompt_target import PromptTarget
from pyrit.common import default_values
from pyrit.memory.memory_models import PromptDataType
from pyrit.prompt_converter import PromptConverter

logger = logging.getLogger(__name__)


class TextToAudio(PromptConverter):
    """
    The TextToAudio takes a prompt and generates a 
    wave file.

    Args:
        speech_region (str): The name of the Azure region.
        speech_key (str): The API key for accessing the service.
        synthesis_language (str): The API key for accessing the service.
        synthesis_voice_name (str): Synthesis voice name, see URL
        https://learn.microsoft.com/en-us/azure/ai-services/speech-service/language-support 
        filename (str): File name to be generated.
    """
    SPEECH_REGION_ENVIRONMENT_VARIABLE: str = "SPEECH_REGION"
    SPEECH_KEY_TOKEN_ENVIRONMENT_VARIABLE: str = "SPEECH_KEY_TOKEN"

    def has_wav_extension(self, file_name):
        return file_name.lower().endswith(".wav")

    def __init__(
        self,
        *,
        speech_region: str = None,
        speech_key: str = None,
        synthesis_language: str = None,
        synthesis_voice_name: str = None,
        filename: str = None,
    ):

        if speech_region is None:
            self.speech_region: str = default_values.get_required_value(
            env_var_name=self.SPEECH_REGION_ENVIRONMENT_VARIABLE, passed_value=speech_region
            )
        else:
            self.speech_region = speech_region

        if speech_key is None:
            self.speech_key: str = default_values.get_required_value(
            env_var_name=self.SPEECH_KEY_TOKEN_ENVIRONMENT_VARIABLE, passed_value=speech_key
            )
        else:
            self.speech_key = speech_key

        if synthesis_language is None:
            self.synthesis_language = "en_US"
        else:
            self.synthesis_language = synthesis_language

        if synthesis_voice_name is None:
            self.synthesis_voice_name = "en-US-AvaNeural"
        else:
            self.synthesis_voice_name = synthesis_voice_name

        #self.output_dir = pathlib.Path(RESULTS_PATH) / "audio"
        if filename is None:
            #self.filename = self.output_dir / "test.wav"
            self.filename = "test.wav"
        else:
            if self.has_wav_extension(filename):
                #self.filename = self.output_dir / filename
                self.filename = filename
            else:
                logger.error("File name for wav file does not contain .wav")
                raise

    def is_supported(self, input_type: PromptDataType) -> bool:
        return input_type == "text"

    # Sending a prompt to create an audio file
    def send_prompt_to_audio(self, prompt):
        if prompt is None:
            logger.error("Prompt was empty")
            raise
        try:
            speech_config = speechsdk.SpeechConfig(subscription=self.speech_key, region=self.speech_region)
            speech_config.speech_synthesis_language = self.synthesis_language
            speech_config.speech_synthesis_voice_name = self.synthesis_voice_name
            audio_config = speechsdk.audio.AudioOutputConfig(filename=self.filename)
            speech_synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=audio_config)
            result = speech_synthesizer.speak_text_async(prompt)
            print(result)
        except Exception as e:
            logger.error(e)
            raise

    async def send_prompt_async(self, prompt):
        if prompt is None:
            logger.error("Prompt was empty")
            raise
        
        try:
            speech_config = speechsdk.SpeechConfig(subscription=self.speech_key, region=self.speech_region)
            speech_config.speech_synthesis_language = self.synthesis_language
            speech_config.speech_synthesis_voice_name = self.synthesis_voice_name
            audio_config = speechsdk.audio.AudioOutputConfig(filename=self.filename)
            speech_synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=audio_config)
            speech_synthesizer.speak_text_async(self.prompt)
        except Exception as e:
            logger.error(e)
            raise

    def convert(self, *, prompt: str, input_type: PromptDataType = "text"):
        """
        Simple converter that converts the prompt to capital letters via a percentage .
        """
        if not self.is_supported(input_type):
            raise ValueError("Input type not supported")
        self.send_prompt_to_audio(prompt)