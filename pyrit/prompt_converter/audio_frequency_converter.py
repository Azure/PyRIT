# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
from typing import Literal, Optional

import io
import numpy as np
from scipy.io import wavfile



from pyrit.models import PromptDataType
from pyrit.models.data_type_serializer import data_serializer_factory
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
    def __init__(self, *,  output_format: AcceptedAudioFormats = "wav", memory: Optional[MemoryInterface] = None) -> None:
        self._output_format = output_format
        self._memory = memory or DuckDBMemory()

    def input_supported(self, input_type: PromptDataType) -> bool:
        return input_type == "audio_path"


    async def convert_async(self, *, prompt: str, input_type: PromptDataType = "audio_path", shift_value: int) -> ConverterResult:
        if not self.input_supported(input_type):
            raise ValueError("Input type not supported")
        try:
            audio_serializer = data_serializer_factory(data_type="audio_path", extension=self._output_format, value=prompt, memory=self._memory)
            audio_bytes = await audio_serializer.read_data()
            bytes_io = io.BytesIO(audio_bytes)
            sample_rate, data = wavfile.read(bytes_io)
            shifted_data = data * np.exp(1j * 2 * np.pi * shift_value * np.arange(len(data)) / sample_rate)
            # Convert the real part of the shifted data to int16
            shifted_data_int16 = shifted_data.real.astype(np.int16)

            # Write the NumPy array to the in-memory buffer as a WAV file
            wavfile.write(bytes_io, sample_rate, shifted_data_int16)

            # Retrieve the WAV bytes
            converted_bytes = bytes_io.getvalue() 

            # Use data serializer to write bytes to wav file
            await audio_serializer.save_data(data=converted_bytes)
            audio_serializer_file = str(audio_serializer.value)
            logger.info(
                    "Speech synthesized for text [{}], and the audio was saved to [{}]".format(
                        prompt, audio_serializer_file
                    )
                )

        except Exception as e:
            logger.error("Failed to convert prompt to audio: %s", str(e))
            raise
        return ConverterResult(output_text=audio_serializer_file, output_type=input_type)