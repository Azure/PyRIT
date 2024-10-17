import os
import pytest
import numpy as np
from unittest import mock
from pyrit.prompt_converter.audio_frequency_converter import AudioFrequencyConverter
from pyrit.models import PromptDataType
from pyrit.prompt_converter import ConverterResult

@pytest.fixture
def mock_wavfile():
    with mock.patch('scipy.io.wavfile.read') as mock_read, \
         mock.patch('scipy.io.wavfile.write') as mock_write:
        
        # Mock the read function to return a sample rate and a numpy array
        sample_rate = 44100
        duration = 1  # in seconds
        t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
        frequency = 440  # Frequency in Hz
        amplitude = np.iinfo(np.int16).max  # Max amplitude for int16
        data = amplitude * np.sin(2 * np.pi * frequency * t)
        mock_read.return_value = (sample_rate, data.astype(np.int16))
        
        yield mock_read, mock_write

@pytest.mark.asyncio
async def test_convert_async(mock_wavfile):
    mock_read, mock_write = mock_wavfile
    converter = AudioFrequencyConverter()
    shift_value = 20000  # Shift frequency by 1000 Hz
    result = await converter.convert_async(prompt="mock_audio_file.wav", input_type="audio_path", shift_value=shift_value)
    
    assert result.output_type == "audio_path"
    assert result.output_text.endswith("_converted.wav")
    # Verify that the write function was called
    assert mock_write.called
    _, args, _ = mock_write.mock_calls[0]
    output_sample_rate, output_data = args[1], args[2]
    sample_rate_original, original_data = mock_read("mock_audio_file.wav")
    assert sample_rate_original == output_sample_rate
    assert not np.array_equal(original_data, output_data)


@pytest.mark.asyncio
async def test_convert_async_invalid_input_type(mock_wavfile):
    converter = AudioFrequencyConverter()
    with pytest.raises(ValueError, match="Input type not supported"):
        await converter.convert_async(prompt="mock_audio_file.wav", input_type="invalid_type", shift_value=1000)
