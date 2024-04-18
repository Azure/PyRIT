# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import logging
import pathlib

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
        speech_region (str): The name of the Azure region.
        speech_key (str): The API key for accessing the service.
        synthesis_language (str): The API key for accessing the service.
        synthesis_voice_name (str): Synthesis voice name, see URL
        https://learn.microsoft.com/en-us/azure/ai-services/speech-service/language-support
        filename (str): File name to be generated.
    """
    SPEECH_REGION_ENVIRONMENT_VARIABLE: str = "SPEECH_REGION"
    SPEECH_KEY_TOKEN_ENVIRONMENT_VARIABLE: str = "SPEECH_KEY_TOKEN"

    def __init__(self, 
            filename: str = "hi.wav",
            speech_region: str = None,
            speech_key: str = None,
            synthesis_language: str = None,
            synthesis_voice_name: str = None,
        ) -> None:
        
        self.filename = filename

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

        self.output_dir = pathlib.Path(RESULTS_PATH) / "audio"
        
        if self.has_wav_extension(self.filename):
            self.filename = filename
        else:
            logger.error("File name for wav file does not contain .wav")
            raise

    def is_supported(self, input_type: PromptDataType) -> bool:
        return input_type == "text"

    def has_wav_extension(self, file_name):
        return file_name.lower().endswith(".wav") 

    def send_prompt_to_audio(self, prompt):
        if prompt is None:
            logger.error("Prompt was empty")
            raise
        try:
            speech_config = speechsdk.SpeechConfig(subscription=self.speech_key, region=self.speech_region)
            speech_config.speech_synthesis_language = self.synthesis_language
            speech_config.speech_synthesis_voice_name = self.synthesis_voice_name
            file_name = str(self.filename)
            file_config = speechsdk.audio.AudioOutputConfig(filename=file_name)
            speech_synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=file_config)
            speech_synthesizer
            result = speech_synthesizer.speak_text_async(prompt).get()
        except Exception as e:
           logger.error(e)
           raise

    def convert(self, *, prompt: str, input_type: PromptDataType = "text") -> str:
        """
        Simple converter that converts the prompt to capital letters via a percentage .
        """
        if not self.is_supported(input_type):
            raise ValueError("Input type not supported")

        self.send_prompt_to_audio(prompt)
        
        return ("The following prompt: \"",prompt,"\" was converted into the audio file ",self.filename)
