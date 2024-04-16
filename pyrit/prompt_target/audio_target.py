# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import logging
import pathlib

# !pip install azure-cognitiveservices-speech
import azure.cognitiveservices.speech as speechsdk

from pyrit.common.path import RESULTS_PATH
from pyrit.prompt_target import PromptTarget

logger = logging.getLogger(__name__)


class AudioTarget(PromptTarget):
    """
    The AudioTarget takes a prompt and generates images
    This class initializes a DALL-E image target

    Args:
        deployment_name (str): The name of the deployment.
        endpoint (str): The endpoint URL for the service.
        api_key (str): The API key for accessing the service.
    """

    def has_wav_extension(self, file_name):
        return file_name.lower().endswith(".wav")

    def __init__(
        self,
        *,
        prompt: str = None,
        speech_region: str = None,
        speech_key: str = None,
        synthesis_language: str = None,
        synthesis_voice_name: str = None,
        filename: str = None,
    ):
        if prompt is None:
            logger.error("Prompt was empty")
            raise
        else:
            self.prompt = prompt

        if speech_region is None:
            logger.error("No region specified")
            raise
        else:
            self.speech_region = speech_region

        if speech_key is None:
            logger.error("No key specified for Speech endpoint")
            raise
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
        if filename is None:
            self.filename = self.output_dir / "test.wav"
        else:
            if self.has_wav_extension(filename):
                self.filename = self.output_dir / filename
            else:
                logger.error("File name for wav file does not contain .wav")
                raise

    # Sending a prompt to create an audio file
    def send_prompt_to_audio(self):
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

    async def send_prompt_async(self):
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
