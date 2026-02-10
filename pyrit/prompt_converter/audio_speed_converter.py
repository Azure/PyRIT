# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import io
import logging
from typing import Literal

import numpy as np
from scipy.io import wavfile
from scipy.interpolate import interp1d

from pyrit.models import PromptDataType, data_serializer_factory
from pyrit.prompt_converter.prompt_converter import ConverterResult, PromptConverter

logger = logging.getLogger(__name__)


class AudioSpeedConverter(PromptConverter):
    """
    Changes the playback speed of an audio file without altering pitch or other audio characteristics.

    A speed_factor > 1.0 speeds up the audio (shorter duration),
    while a speed_factor < 1.0 slows it down (longer duration).
    The converter resamples the audio signal using interpolation so that the
    sample rate, bit depth, and number of channels remain unchanged.
    """

    SUPPORTED_INPUT_TYPES = ("audio_path",)
    SUPPORTED_OUTPUT_TYPES = ("audio_path",)

    #: Accepted audio formats for conversion.
    AcceptedAudioFormats = Literal["wav"]

    def __init__(
        self,
        *,
        output_format: AcceptedAudioFormats = "wav",
        speed_factor: float = 1.5,
    ) -> None:
        """
        Initialize the converter with the specified output format and speed factor.

        Args:
            output_format (str): The format of the audio file, defaults to "wav".
            speed_factor (float): The factor by which to change the speed.
                Values > 1.0 speed up the audio, values < 1.0 slow it down.
                Must be greater than 0. Defaults to 1.5.

        Raises:
            ValueError: If speed_factor is not positive.
        """
        if speed_factor <= 0:
            raise ValueError("speed_factor must be greater than 0.")
        self._output_format = output_format
        self._speed_factor = speed_factor

    async def convert_async(self, *, prompt: str, input_type: PromptDataType = "audio_path") -> ConverterResult:
        """
        Convert the given audio file by changing its playback speed.

        The audio is resampled via interpolation so that the output has a different
        number of samples (and therefore a different duration) while keeping the
        original sample rate. This preserves the pitch and tonal qualities of the audio.

        Args:
            prompt (str): File path to the audio file to be converted.
            input_type (PromptDataType): The type of input data.

        Returns:
            ConverterResult: The result containing the converted audio file path.

        Raises:
            ValueError: If the input type is not supported.
            Exception: If there is an error during the conversion process.
        """
        if not self.input_supported(input_type):
            raise ValueError("Input type not supported")
        try:
            # Create serializer to read audio data
            audio_serializer = data_serializer_factory(
                category="prompt-memory-entries", data_type="audio_path", extension=self._output_format, value=prompt
            )
            audio_bytes = await audio_serializer.read_data()

            # Read the audio file bytes and process the data
            bytes_io = io.BytesIO(audio_bytes)
            sample_rate, data = wavfile.read(bytes_io)
            original_dtype = data.dtype

            # Handle both mono and multi-channel audio
            if data.ndim == 1:
                # Mono audio
                num_samples = len(data)
                new_num_samples = int(num_samples / self._speed_factor)

                # Create interpolation function and resample
                original_indices = np.arange(num_samples)
                new_indices = np.linspace(0, num_samples - 1, new_num_samples)
                interpolator = interp1d(original_indices, data.astype(np.float64), kind="linear")
                resampled_data = interpolator(new_indices).astype(original_dtype)
            else:
                # Multi-channel audio (e.g., stereo)
                num_samples = data.shape[0]
                new_num_samples = int(num_samples / self._speed_factor)

                original_indices = np.arange(num_samples)
                new_indices = np.linspace(0, num_samples - 1, new_num_samples)

                channels = []
                for ch in range(data.shape[1]):
                    interpolator = interp1d(original_indices, data[:, ch].astype(np.float64), kind="linear")
                    channels.append(interpolator(new_indices))
                resampled_data = np.column_stack(channels).astype(original_dtype)

            # Write the resampled data as a new WAV file
            output_bytes_io = io.BytesIO()
            wavfile.write(output_bytes_io, sample_rate, resampled_data)

            # Save the converted bytes using the serializer
            converted_bytes = output_bytes_io.getvalue()
            await audio_serializer.save_data(data=converted_bytes)
            audio_serializer_file = str(audio_serializer.value)
            logger.info(
                "Audio speed changed by factor %.2f for [%s], and the audio was saved to [%s]",
                self._speed_factor,
                prompt,
                audio_serializer_file,
            )

        except Exception as e:
            logger.error("Failed to convert audio speed: %s", str(e))
            raise
        return ConverterResult(output_text=audio_serializer_file, output_type=input_type)
