# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import os
import logging
import time
import azure.cognitiveservices.speech as speechsdk

from pyrit.common import default_values
from pyrit.models import PromptDataType
from pyrit.prompt_converter import ConverterResult, PromptConverter

logger = logging.getLogger(__name__)


class AzureSpeechAudioToTextConverter(PromptConverter):
    """
    The AzureSpeechAudioTextConverter takes a .wav file and transcribes it into text.
    https://learn.microsoft.com/en-us/azure/ai-services/speech-service/speech-to-text
    Args:
        azure_speech_region (str): The name of the Azure region.
        azure_speech_key (str): The API key for accessing the service.
        recognition_language (str): Recognition voice language. Defaults to "en-US".
            For more on supported languages, see the following link:
            https://learn.microsoft.com/en-us/azure/ai-services/speech-service/language-support
    """

    AZURE_SPEECH_REGION_ENVIRONMENT_VARIABLE: str = "AZURE_SPEECH_REGION"
    AZURE_SPEECH_KEY_ENVIRONMENT_VARIABLE: str = "AZURE_SPEECH_KEY"

    def __init__(
        self,
        azure_speech_region: str = None,
        azure_speech_key: str = None,
        recognition_language: str = "en-US",
    ) -> None:

        self._azure_speech_region: str = default_values.get_required_value(
            env_var_name=self.AZURE_SPEECH_REGION_ENVIRONMENT_VARIABLE, passed_value=azure_speech_region
        )

        self._azure_speech_key: str = default_values.get_required_value(
            env_var_name=self.AZURE_SPEECH_KEY_ENVIRONMENT_VARIABLE, passed_value=azure_speech_key
        )

        self._recognition_language = recognition_language

    def input_supported(self, input_type: PromptDataType) -> bool:
        return input_type == "text"

    async def convert_async(self, *, prompt: str, input_type: PromptDataType = "text") -> ConverterResult:
        """
        Converter that transcribes audio to text.

        Args:
            prompt (str): File path to audio file
            input_type (PromptDataType): Type of data
        Returns:
            ConverterResult: The transcribed text as a ConverterResult Object
        """
        if not self.input_supported(input_type):
            raise ValueError("Input type not supported")

        if not os.path.exists(prompt):
            raise ValueError("File path does not exist. Please provide valid audio file path.")

        if not prompt.endswith(".wav"):
            raise ValueError("Please provide a .wav audio file. Compressed formats are not currently supported.")

        try:
            transcript = self.recognize_audio(prompt)
        except Exception as e:
            logger.error("Failed to convert audio file to text: %s", str(e))
            raise
        return ConverterResult(output_text=transcript, output_type="text")

    def recognize_audio(self, audio_file: str) -> str:
        """
        Recognize audio file and return transcribed text.

        Args:
            audio_file (str): File path to audio file
        Returns:
            str: Transcribed text
        """
        speech_config = speechsdk.SpeechConfig(
            subscription=self._azure_speech_key,
            region=self._azure_speech_region,
        )
        speech_config.speech_recognition_language = self._recognition_language

        audio_config = speechsdk.audio.AudioConfig(filename=audio_file)
        # Instantiate a speech recognizer object
        speech_recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)
        # Create an empty list to store recognized text
        transcribed_text = []
        done = False

        def transcript_cb(evt: speechsdk.SessionEventArgs) -> None:
            """
            Callback function that appends transcribed text upon receiving a "recognized" event

            Args:
                evt (SessionEventArgs): event
            """
            logging.info("RECOGNIZED: {}".format(evt.result.text))
            transcribed_text.append(evt.result.text)

        def stop_cb(evt: speechsdk.SessionEventArgs) -> None:
            """
            Callback function that stops continuous recognition upon receiving an event 'evt'

            Args:
                evt (SessionEventArgs): event
            """
            logging.info("CLOSING on {}".format(evt))
            speech_recognizer.stop_continuous_recognition_async()
            nonlocal done
            done = True
            if evt.result.reason == speechsdk.ResultReason.Canceled:
                cancellation_details = evt.result.cancellation_details
                logging.info("Speech recognition canceled: {}".format(cancellation_details.reason))
                if cancellation_details.reason == speechsdk.CancellationReason.Error:
                    logger.error("Error details: {}".format(cancellation_details.error_details))
                elif cancellation_details.reason == speechsdk.CancellationReason.EndOfStream:
                    logging.info("End of audio stream detected.")

        # Connect callbacks to the events fired by the speech recognizer
        speech_recognizer.recognized.connect(transcript_cb)
        speech_recognizer.recognizing.connect(lambda evt: logging.info("RECOGNIZING: {}".format(evt)))
        speech_recognizer.recognized.connect(lambda evt: logging.info("RECOGNIZED: {}".format(evt)))
        speech_recognizer.session_started.connect(lambda evt: logging.info("SESSION STARTED: {}".format(evt)))
        speech_recognizer.session_stopped.connect(lambda evt: logging.info("SESSION STOPPED: {}".format(evt)))

        # Stop continuous recognition when stopped or canceled event is fired
        speech_recognizer.canceled.connect(stop_cb)
        speech_recognizer.session_stopped.connect(stop_cb)

        # Start continuous recognition
        speech_recognizer.start_continuous_recognition_async()
        while not done:
            time.sleep(0.5)

        return "".join(transcribed_text)
