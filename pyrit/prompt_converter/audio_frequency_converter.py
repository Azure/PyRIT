import logging
from typing import Literal, Optional

import numpy as np
from scipy.io import wavfile



from pyrit.models import PromptDataType
from pyrit.prompt_converter import PromptConverter,  ConverterResult
from pyrit.memory import MemoryInterface, DuckDBMemory

logger = logging.getLogger(__name__)


class AudioFrequencyConverter(PromptConverter):
    """
    The AudioFrequencyConverter takes an audio file and shifts its frequency, by default it will shift it above human range (>20kHz).
    Args:
        audio_file_path (str): The path to the audio file to be converted.
        converted_file_path (str): The path to the new converted audio file.
        audio_format (str): The format of the audio file. Defaults to "wav".
    """
    AcceptedAudioFormats = Literal["wav"]
    def __init__(self, *, audio_format: AcceptedAudioFormats = "wav") -> None:
        self._audio_format = audio_format


    def input_supported(self, input_type: PromptDataType) -> bool:
        return input_type == "audio_path"

    async def convert_async(self, *, prompt: str, input_type: PromptDataType = "audio_path", shift_value: int) -> ConverterResult:
        if not self.input_supported(input_type):
            raise ValueError("Input type not supported")
        try:
            sample_rate, data = wavfile.read(prompt)
            shifted_data = data * np.exp(1j * 2 * np.pi * shift_value * np.arange(len(data)) / sample_rate)
            converted_file_path = prompt.replace('.wav', '_converted.wav')
            wavfile.write(converted_file_path, sample_rate, shifted_data.real.astype(np.int16))
            logger.info(
                    "Speech synthesized for text [{}], and the audio was saved to [{}]".format(
                        prompt, converted_file_path
                    )
                )

        except Exception as e:
            logger.error("Failed to convert prompt to audio: %s", str(e))
            raise
        return ConverterResult(output_text=converted_file_path, output_type=input_type)