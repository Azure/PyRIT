# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


import pytest
import os
import tempfile

import numpy as np
from scipy.io import wavfile

from pyrit.memory import DuckDBMemory, CentralMemory
from pyrit.prompt_converter.audio_frequency_converter import AudioFrequencyConverter


@pytest.fixture(scope="function")
def set_duckdb_in_memory():
    duckdb_in_memory = DuckDBMemory(db_path=":memory:")
    CentralMemory.set_memory_instance(duckdb_in_memory)


@pytest.mark.asyncio
async def test_convert_async_success(set_duckdb_in_memory):

    # Simulate WAV data
    sample_rate = 44100
    mock_audio_data = np.random.randint(-32768, 32767, size=(100,), dtype=np.int16)

    # Create a temporary file for the WAV file
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_wav_file:
        original_wav_path = temp_wav_file.name
        wavfile.write(original_wav_path, sample_rate, mock_audio_data)

    converter = AudioFrequencyConverter(shift_value=20000)

    # Call the convert_async method with the temporary WAV file path
    result = await converter.convert_async(prompt=original_wav_path)

    assert os.path.exists(result.output_text)
    assert isinstance(result.output_text, str)

    # Clean up the original WAV file after the test
    os.remove(original_wav_path)

    # Optionally, clean up any created files in result (if applicable)
    if os.path.exists(result.output_text):
        os.remove(result.output_text)


@pytest.mark.asyncio
async def test_convert_async_file_not_found(set_duckdb_in_memory):

    # Create an instance of the converter
    converter = AudioFrequencyConverter(shift_value=20000)

    prompt = "non_existent_file.wav"

    # Ensure that an exception is raised when trying to convert a non-existent file
    with pytest.raises(FileNotFoundError):
        await converter.convert_async(prompt=prompt)
