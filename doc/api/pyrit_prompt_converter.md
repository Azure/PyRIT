# pyrit.prompt_converter

Prompt converters for transforming prompts before sending them to targets in red teaming workflows.

Converters are organized into categories: Text-to-Text (encoding, obfuscation, translation, variation),
Audio (text-to-audio, audio-to-text, audio-to-audio), Image (text-to-image, image-to-image),
Video (image-to-video), File (text-to-PDF/URL), Selective Converting (partial prompt transformation),
and Human-in-the-Loop (interactive review). Converters can be stacked together to create complex
transformation pipelines for testing AI system robustness.

## Functions

### get_converter_modalities

```python
get_converter_modalities() → list[tuple[str, list[PromptDataType], list[PromptDataType]]]
```

Retrieve a list of all converter classes and their supported input/output modalities
by reading the SUPPORTED_INPUT_TYPES and SUPPORTED_OUTPUT_TYPES class attributes.

**Returns:**

- `list[tuple[str, list[PromptDataType], list[PromptDataType]]]` — list[tuple[str, list[PromptDataType], list[PromptDataType]]]: A sorted list of tuples containing:
- Converter class name (str)
- List of supported input modalities (list[PromptDataType])
- List of supported output modalities (list[PromptDataType])
- `list[tuple[str, list[PromptDataType], list[PromptDataType]]]` — Sorted by input modality, then output modality, then converter name.

## `class AddImageTextConverter(PromptConverter)`

Adds a string to an image and wraps the text into multiple lines if necessary.

This class is similar to :class:`AddTextImageConverter` except
we pass in an image file path as an argument to the constructor as opposed to text.

**Constructor Parameters:**

| Parameter | Type | Description |
|---|---|---|
| `img_to_add` | `str` | File path of image to add text to. |
| `font_name` | `str` | Path of font to use. Must be a TrueType font (.ttf). Defaults to "helvetica.ttf". Defaults to `'helvetica.ttf'`. |
| `color` | `tuple` | Color to print text in, using RGB values. Defaults to (0, 0, 0). Defaults to `(0, 0, 0)`. |
| `font_size` | `float` | Size of font to use. Defaults to 15. Defaults to `15`. |
| `x_pos` | `int` | X coordinate to place text in (0 is left most). Defaults to 10. Defaults to `10`. |
| `y_pos` | `int` | Y coordinate to place text in (0 is upper most). Defaults to 10. Defaults to `10`. |

**Methods:**

#### convert_async

```python
convert_async(prompt: str, input_type: PromptDataType = 'text') → ConverterResult
```

Convert the given prompt by adding it as text to the image.

| Parameter | Type | Description |
|---|---|---|
| `prompt` | `str` | The text to be added to the image. |
| `input_type` | `PromptDataType` | The type of input data. Defaults to `'text'`. |

**Returns:**

- `ConverterResult` — The result containing path to the updated image.

**Raises:**

- `ValueError` — If the input type is not supported.

## `class AddImageVideoConverter(PromptConverter)`

Adds an image to a video at a specified position.

Currently the image is placed in the whole video, not at a specific timepoint.

**Constructor Parameters:**

| Parameter | Type | Description |
|---|---|---|
| `video_path` | `str` | File path of video to add image to. |
| `output_path` | `(str, Optional)` | File path of output video. Defaults to None. Defaults to `None`. |
| `img_position` | `tuple` | Position to place image in video. Defaults to (10, 10). Defaults to `(10, 10)`. |
| `img_resize_size` | `tuple` | Size to resize image to. Defaults to (500, 500). Defaults to `(500, 500)`. |

**Methods:**

#### convert_async

```python
convert_async(prompt: str, input_type: PromptDataType = 'image_path') → ConverterResult
```

Convert the given prompt (image) by adding it to a video.

| Parameter | Type | Description |
|---|---|---|
| `prompt` | `str` | The image path to be added to the video. |
| `input_type` | `PromptDataType` | The type of input data. Defaults to `'image_path'`. |

**Returns:**

- `ConverterResult` — The result containing filename of the converted video.

**Raises:**

- `ValueError` — If the input type is not supported.

## `class AddTextImageConverter(PromptConverter)`

Adds a string to an image and wraps the text into multiple lines if necessary.

This class is similar to :class:`AddImageTextConverter` except
we pass in text as an argument to the constructor as opposed to an image file path.

**Constructor Parameters:**

| Parameter | Type | Description |
|---|---|---|
| `text_to_add` | `str` | Text to add to an image. Defaults to empty string. |
| `font_name` | `str` | Path of font to use. Must be a TrueType font (.ttf). Defaults to "helvetica.ttf". Defaults to `'helvetica.ttf'`. |
| `color` | `tuple` | Color to print text in, using RGB values. Defaults to (0, 0, 0). Defaults to `(0, 0, 0)`. |
| `font_size` | `float` | Size of font to use. Defaults to 15. Defaults to `15`. |
| `x_pos` | `int` | X coordinate to place text in (0 is left most). Defaults to 10. Defaults to `10`. |
| `y_pos` | `int` | Y coordinate to place text in (0 is upper most). Defaults to 10. Defaults to `10`. |

**Methods:**

#### convert_async

```python
convert_async(prompt: str, input_type: PromptDataType = 'image_path') → ConverterResult
```

Convert the given prompt (image) by adding text to it.

| Parameter | Type | Description |
|---|---|---|
| `prompt` | `str` | The image file path to which text will be added. |
| `input_type` | `PromptDataType` | The type of input data. Defaults to `'image_path'`. |

**Returns:**

- `ConverterResult` — The result containing path to the updated image.

**Raises:**

- `ValueError` — If the input type is not supported.

## `class AllWordsSelectionStrategy(WordSelectionStrategy)`

Selects all words (default strategy).

**Methods:**

#### `select_words(words: list[str]) → list[int]`

Select all words.

| Parameter | Type | Description |
|---|---|---|
| `words` | `List[str]` | The list of words to select from. |

**Returns:**

- `list[int]` — List[int]: All word indices.

## `class AsciiArtConverter(PromptConverter)`

Uses the `art` package to convert text into ASCII art.

**Constructor Parameters:**

| Parameter | Type | Description |
|---|---|---|
| `font` | `str` | The font to use for ASCII art. Defaults to "rand" which selects a random font. Defaults to `'rand'`. |

**Methods:**

#### convert_async

```python
convert_async(prompt: str, input_type: PromptDataType = 'text') → ConverterResult
```

Convert the given prompt into ASCII art.

| Parameter | Type | Description |
|---|---|---|
| `prompt` | `str` | The prompt to be converted. |
| `input_type` | `PromptDataType` | The type of input data. Defaults to `'text'`. |

**Returns:**

- `ConverterResult` — The result containing the ASCII art representation of the prompt.

**Raises:**

- `ValueError` — If the input type is not supported.

## `class AsciiSmugglerConverter(SmugglerConverter)`

Implements encoding and decoding using Unicode Tags.

If 'control' is True, the encoded output is wrapped with:
    - U+E0001 (start control tag)
    - U+E007F (end control tag)

Replicates the functionality detailed in the following blog post:
https://embracethered.com/blog/posts/2024/hiding-and-finding-text-with-unicode-tags/

**Constructor Parameters:**

| Parameter | Type | Description |
|---|---|---|
| `action` | `Literal['encode', 'decode']` | The action to perform. Defaults to `'encode'`. |
| `unicode_tags` | `bool` | Whether to add Unicode tags during encoding. Defaults to `False`. |

**Methods:**

#### `decode_message(message: str) → str`

Decode a message encoded with Unicode Tags.

For each character in the Unicode Tags range, subtracts 0xE0000.
Skips control tags if present.

| Parameter | Type | Description |
|---|---|---|
| `message` | `str` | The encoded message. |

**Returns:**

- `str` — The decoded message.

#### `encode_message(message: str) → tuple[str, str]`

Encode the message using Unicode Tags.

Each ASCII printable character (0x20-0x7E) is mapped to a corresponding
Unicode Tag (by adding 0xE0000). If control mode is enabled, wraps the output.

| Parameter | Type | Description |
|---|---|---|
| `message` | `str` | The message to encode. |

**Returns:**

- `tuple[str, str]` — Tuple[str, str]: A tuple with a summary of code points and the encoded message.

## `class AskToDecodeConverter(PromptConverter)`

Wraps encoded text with prompts that ask a target to decode it.

This converter takes encoded text (e.g., Base64, ROT13, Morse code) and wraps it
in various prompt templates that request decoding. The prompts can be generic
("Decode the following text:") or encoding-specific ("Base64 encoded string:").
This is useful for testing whether AI systems will decode potentially harmful
encoded content when explicitly asked.

Credit to Garak: https://github.com/NVIDIA/garak/blob/main/garak/probes/encoding.py

**Constructor Parameters:**

| Parameter | Type | Description |
|---|---|---|
| `template` | `str` | Custom template for conversion. Should include {encoded_text} placeholder and optionally {encoding_name} placeholder. If None, a random template is selected. Defaults to None. Defaults to `None`. |
| `encoding_name` | `str` | Name of the encoding scheme (e.g., "Base64", "ROT13", "Morse"). Used in encoding_name_templates to provide context about the encoding type. Defaults to empty string. Defaults to `'cipher'`. |

**Methods:**

#### convert_async

```python
convert_async(prompt: str, input_type: PromptDataType = 'text') → ConverterResult
```

Convert the given encoded text by wrapping it with a decoding request prompt.

| Parameter | Type | Description |
|---|---|---|
| `prompt` | `str` | The encoded text to be wrapped with a decoding request. |
| `input_type` | `PromptDataType` | Type of input data. Defaults to "text". Defaults to `'text'`. |

**Returns:**

- `ConverterResult` — The result containing the converted prompt.

**Raises:**

- `ValueError` — If the input type is not supported (only "text" is supported).

## `class AtbashConverter(PromptConverter)`

'Hello 123' would encode to 'Svool 876'.

**Constructor Parameters:**

| Parameter | Type | Description |
|---|---|---|
| `append_description` | `bool` | If True, appends plaintext "expert" text to the prompt. This includes instructions to only communicate using the cipher, a description of the cipher, and an example encoded using the cipher. Defaults to `False`. |

**Methods:**

#### convert_async

```python
convert_async(prompt: str, input_type: PromptDataType = 'text') → ConverterResult
```

Convert the given prompt using the Atbash cipher.

| Parameter | Type | Description |
|---|---|---|
| `prompt` | `str` | The prompt to be converted. |
| `input_type` | `PromptDataType` | The type of input data. Defaults to `'text'`. |

**Returns:**

- `ConverterResult` — The result containing the encoded prompt.

**Raises:**

- `ValueError` — If the input type is not supported.

## `class AudioEchoConverter(PromptConverter)`

Adds an echo effect to an audio file.

The echo is created by mixing a delayed, attenuated copy of the signal back
into the original. The delay and decay parameters control the timing and
loudness of the echo respectively. Sample rate, bit depth, and channel
count are preserved.

**Constructor Parameters:**

| Parameter | Type | Description |
|---|---|---|
| `output_format` | `str` | The format of the audio file, defaults to "wav". Defaults to `'wav'`. |
| `delay` | `float` | The echo delay in seconds. Must be greater than 0. Defaults to 0.3. Defaults to `0.3`. |
| `decay` | `float` | The decay factor for the echo (0.0 to 1.0). A value of 0.0 means no echo, 1.0 means the echo is as loud as the original. Must be between 0 and 1 (exclusive of both). Defaults to 0.5. Defaults to `0.5`. |

**Methods:**

#### convert_async

```python
convert_async(prompt: str, input_type: PromptDataType = 'audio_path') → ConverterResult
```

Convert the given audio file by adding an echo effect.

| Parameter | Type | Description |
|---|---|---|
| `prompt` | `str` | File path to the audio file to be converted. |
| `input_type` | `PromptDataType` | The type of input data. Defaults to `'audio_path'`. |

**Returns:**

- `ConverterResult` — The result containing the converted audio file path.

**Raises:**

- `ValueError` — If the input type is not supported.
- `Exception` — If there is an error during the conversion process.

## `class AudioFrequencyConverter(PromptConverter)`

Shifts the frequency of an audio file by a specified value.
By default, it will shift it above the human hearing range (=20 kHz).

**Constructor Parameters:**

| Parameter | Type | Description |
|---|---|---|
| `output_format` | `str` | The format of the audio file, defaults to "wav". Defaults to `'wav'`. |
| `shift_value` | `int` | The value by which the frequency will be shifted, defaults to 20000 Hz. Defaults to `20000`. |

**Methods:**

#### convert_async

```python
convert_async(prompt: str, input_type: PromptDataType = 'audio_path') → ConverterResult
```

Convert the given audio file by shifting its frequency.

| Parameter | Type | Description |
|---|---|---|
| `prompt` | `str` | File path to the audio file to be converted. |
| `input_type` | `PromptDataType` | The type of input data. Defaults to `'audio_path'`. |

**Returns:**

- `ConverterResult` — The result containing the audio file path.

**Raises:**

- `ValueError` — If the input type is not supported.
- `Exception` — If there is an error during the conversion process.

## `class AudioSpeedConverter(PromptConverter)`

Changes the playback speed of an audio file without altering pitch or other audio characteristics.

A speed_factor > 1.0 speeds up the audio (shorter duration),
while a speed_factor < 1.0 slows it down (longer duration).
The converter resamples the audio signal using interpolation so that the
sample rate, bit depth, and number of channels remain unchanged.

**Constructor Parameters:**

| Parameter | Type | Description |
|---|---|---|
| `output_format` | `str` | The format of the audio file, defaults to "wav". Defaults to `'wav'`. |
| `speed_factor` | `float` | The factor by which to change the speed. Values > 1.0 speed up the audio, values < 1.0 slow it down. Must be greater than 0 and at most 100. Defaults to 1.5. Defaults to `1.5`. |

**Methods:**

#### convert_async

```python
convert_async(prompt: str, input_type: PromptDataType = 'audio_path') → ConverterResult
```

Convert the given audio file by changing its playback speed.

The audio is resampled via interpolation so that the output has a different
number of samples (and therefore a different duration) while keeping the
original sample rate. This preserves the pitch and tonal qualities of the audio.

| Parameter | Type | Description |
|---|---|---|
| `prompt` | `str` | File path to the audio file to be converted. |
| `input_type` | `PromptDataType` | The type of input data. Defaults to `'audio_path'`. |

**Returns:**

- `ConverterResult` — The result containing the converted audio file path.

**Raises:**

- `ValueError` — If the input type is not supported.
- `Exception` — If there is an error during the conversion process.

## `class AudioVolumeConverter(PromptConverter)`

Changes the volume of an audio file by scaling the amplitude.

A volume_factor > 1.0 increases the volume (louder),
while a volume_factor < 1.0 decreases it (quieter).
A volume_factor of 1.0 leaves the audio unchanged.
The converter scales all audio samples by the given factor and clips
the result to the valid range for the original data type.
Sample rate, bit depth, and number of channels are preserved.

**Constructor Parameters:**

| Parameter | Type | Description |
|---|---|---|
| `output_format` | `str` | The format of the audio file, defaults to "wav". Defaults to `'wav'`. |
| `volume_factor` | `float` | The factor by which to scale the volume. Values > 1.0 increase volume, values < 1.0 decrease volume. Must be greater than 0. Defaults to 1.5. Defaults to `1.5`. |

**Methods:**

#### convert_async

```python
convert_async(prompt: str, input_type: PromptDataType = 'audio_path') → ConverterResult
```

Convert the given audio file by changing its volume.

The audio samples are scaled by the volume factor. For integer audio
formats the result is clipped to prevent overflow.

| Parameter | Type | Description |
|---|---|---|
| `prompt` | `str` | File path to the audio file to be converted. |
| `input_type` | `PromptDataType` | The type of input data. Defaults to `'audio_path'`. |

**Returns:**

- `ConverterResult` — The result containing the converted audio file path.

**Raises:**

- `ValueError` — If the input type is not supported.
- `Exception` — If there is an error during the conversion process.

## `class AudioWhiteNoiseConverter(PromptConverter)`

Adds white noise to an audio file.

White noise is generated and mixed into the original signal at a level
controlled by the noise_scale parameter. The output preserves the original
sample rate, bit depth, channel count, and number of samples.

**Constructor Parameters:**

| Parameter | Type | Description |
|---|---|---|
| `output_format` | `str` | The format of the audio file, defaults to "wav". Defaults to `'wav'`. |
| `noise_scale` | `float` | Controls the amplitude of the added noise, expressed as a fraction of the signal's maximum possible value. For int16 audio the noise amplitude will be noise_scale * 32767. Must be greater than 0 and at most 1.0. Defaults to 0.02. Defaults to `0.02`. |

**Methods:**

#### convert_async

```python
convert_async(prompt: str, input_type: PromptDataType = 'audio_path') → ConverterResult
```

Convert the given audio file by adding white noise.

| Parameter | Type | Description |
|---|---|---|
| `prompt` | `str` | File path to the audio file to be converted. |
| `input_type` | `PromptDataType` | The type of input data. Defaults to `'audio_path'`. |

**Returns:**

- `ConverterResult` — The result containing the converted audio file path.

**Raises:**

- `ValueError` — If the input type is not supported.
- `Exception` — If there is an error during the conversion process.

## `class AzureSpeechAudioToTextConverter(PromptConverter)`

Transcribes a .wav audio file into text using Azure AI Speech service.

https://learn.microsoft.com/en-us/azure/ai-services/speech-service/speech-to-text

**Constructor Parameters:**

| Parameter | Type | Description |
|---|---|---|
| `azure_speech_region` | `(str, Optional)` | The name of the Azure region. Defaults to `None`. |
| `azure_speech_key` | `(str, Optional)` | The API key for accessing the service (if not using Entra ID auth). Defaults to `None`. |
| `azure_speech_resource_id` | `(str, Optional)` | The resource ID for accessing the service when using Entra ID auth. This can be found by selecting 'Properties' in the 'Resource Management' section of your Azure Speech resource in the Azure portal. Defaults to `None`. |
| `use_entra_auth` | `bool` | Whether to use Entra ID authentication. If True, azure_speech_resource_id must be provided. If False, azure_speech_key must be provided. Defaults to False. Defaults to `False`. |
| `recognition_language` | `str` | Recognition voice language. Defaults to "en-US". For more on supported languages, see the following link: https://learn.microsoft.com/en-us/azure/ai-services/speech-service/language-support Defaults to `'en-US'`. |

**Methods:**

#### convert_async

```python
convert_async(prompt: str, input_type: PromptDataType = 'audio_path') → ConverterResult
```

Convert the given audio file into its text representation.

| Parameter | Type | Description |
|---|---|---|
| `prompt` | `str` | File path to the audio file to be transcribed. |
| `input_type` | `PromptDataType` | The type of input data. Defaults to `'audio_path'`. |

**Returns:**

- `ConverterResult` — The result containing the transcribed text.

**Raises:**

- `ValueError` — If the input type is not supported or if the provided file is not a .wav file.

#### `recognize_audio(audio_bytes: bytes) → str`

Recognize audio file and return transcribed text.

| Parameter | Type | Description |
|---|---|---|
| `audio_bytes` | `bytes` | Audio bytes input. |

**Returns:**

- `str` — Transcribed text.

**Raises:**

- `ModuleNotFoundError` — If the azure.cognitiveservices.speech module is not installed.

#### `stop_cb(evt: Any, recognizer: Any) → None`

Stop continuous recognition upon receiving an event 'evt'.

| Parameter | Type | Description |
|---|---|---|
| `evt` | `speechsdk.SpeechRecognitionEventArgs` | Event. |
| `recognizer` | `speechsdk.SpeechRecognizer` | Speech recognizer object. |

**Raises:**

- `ModuleNotFoundError` — If the azure.cognitiveservices.speech module is not installed.

#### `transcript_cb(evt: Any, transcript: list[str]) → None`

Append transcribed text upon receiving a "recognized" event.

| Parameter | Type | Description |
|---|---|---|
| `evt` | `speechsdk.SpeechRecognitionEventArgs` | Event. |
| `transcript` | `list` | List to store transcribed text. |

## `class AzureSpeechTextToAudioConverter(PromptConverter)`

Generates a wave file from a text prompt using Azure AI Speech service.

https://learn.microsoft.com/en-us/azure/ai-services/speech-service/text-to-speech

**Constructor Parameters:**

| Parameter | Type | Description |
|---|---|---|
| `azure_speech_region` | `(str, Optional)` | The name of the Azure region. Defaults to `None`. |
| `azure_speech_key` | `(str, Optional)` | The API key for accessing the service (only if you're not using Entra authentication). Defaults to `None`. |
| `azure_speech_resource_id` | `(str, Optional)` | The resource ID for accessing the service when using Entra ID auth. This can be found by selecting 'Properties' in the 'Resource Management' section of your Azure Speech resource in the Azure portal. Defaults to `None`. |
| `use_entra_auth` | `bool` | Whether to use Entra ID authentication. If True, azure_speech_resource_id must be provided. If False, azure_speech_key must be provided. Defaults to False. Defaults to `False`. |
| `synthesis_language` | `str` | Synthesis voice language. Defaults to `'en_US'`. |
| `synthesis_voice_name` | `str` | Synthesis voice name, see URL. For more details see the following link for synthesis language and synthesis voice: https://learn.microsoft.com/en-us/azure/ai-services/speech-service/language-support Defaults to `'en-US-AvaNeural'`. |
| `output_format` | `str` | Either wav or mp3. Must match the file prefix. Defaults to `'wav'`. |

**Methods:**

#### convert_async

```python
convert_async(prompt: str, input_type: PromptDataType = 'text') → ConverterResult
```

Convert the given text prompt into its audio representation.

| Parameter | Type | Description |
|---|---|---|
| `prompt` | `str` | The text prompt to be converted into audio. |
| `input_type` | `PromptDataType` | The type of input data. Defaults to `'text'`. |

**Returns:**

- `ConverterResult` — The result containing the audio file path.

**Raises:**

- `ModuleNotFoundError` — If the ``azure.cognitiveservices.speech`` module is not installed.
- `RuntimeError` — If there is an error during the speech synthesis process.
- `ValueError` — If the input type is not supported or if the prompt is empty.

## `class Base2048Converter(PromptConverter)`

Converter that encodes text to base2048 format.

This converter takes input text and converts it to base2048 encoding,
which uses 2048 different Unicode characters to represent binary data.
This can be useful for obfuscating text or testing how systems
handle encoded Unicode content.

**Methods:**

#### convert_async

```python
convert_async(prompt: str, input_type: PromptDataType = 'text') → ConverterResult
```

Convert the given prompt to base2048 encoding.

| Parameter | Type | Description |
|---|---|---|
| `prompt` | `str` | The prompt to be converted. |
| `input_type` | `PromptDataType` | Type of data, unused for this converter. Defaults to `'text'`. |

**Returns:**

- `ConverterResult` — The converted text representation of the original prompt in base2048.

**Raises:**

- `ValueError` — If the input type is not supported.

## `class Base64Converter(PromptConverter)`

Converter that encodes text to base64 format.

This converter takes input text and converts it to base64 encoding,
which can be useful for obfuscating text or testing how systems
handle encoded content.

**Constructor Parameters:**

| Parameter | Type | Description |
|---|---|---|
| `encoding_func` | `EncodingFunc` | The base64 encoding function to use. Defaults to "b64encode". Defaults to `'b64encode'`. |

**Methods:**

#### convert_async

```python
convert_async(prompt: str, input_type: PromptDataType = 'text') → ConverterResult
```

Convert the given prompt to base64 encoding.

| Parameter | Type | Description |
|---|---|---|
| `prompt` | `str` | The prompt to be converted. |
| `input_type` | `PromptDataType` | Type of data, unused for this converter. Defaults to `'text'`. |

**Returns:**

- `ConverterResult` — The converted text representation of the original prompt in base64.

**Raises:**

- `ValueError` — If the input type is not supported.

## `class BinAsciiConverter(WordLevelConverter)`

Converts text to various binary-to-ASCII encodings.

Supports hex, quoted-printable, and UUencode formats.

**Constructor Parameters:**

| Parameter | Type | Description |
|---|---|---|
| `encoding_func` | `str` | The encoding function to use. Options: "hex", "quoted-printable", "UUencode". Defaults to "hex". Defaults to `'hex'`. |
| `word_selection_strategy` | `Optional[WordSelectionStrategy]` | Strategy for selecting which words to convert. If None, all words will be converted. Defaults to `None`. |
| `word_split_separator` | `Optional[str]` | Separator used to split words in the input text. Defaults to " ". Defaults to `' '`. |

**Methods:**

#### `convert_word_async(word: str) → str`

Convert a word using the specified encoding function.

| Parameter | Type | Description |
|---|---|---|
| `word` | `str` | The word to encode. |

**Returns:**

- `str` — The encoded word.

**Raises:**

- `ValueError` — If an unsupported encoding function is encountered.

#### `join_words(words: list[str]) → str`

Join words appropriately based on the encoding type and selection mode.

| Parameter | Type | Description |
|---|---|---|
| `words` | `list[str]` | The list of encoded words to join. |

**Returns:**

- `str` — The joined string.

## `class BinaryConverter(WordLevelConverter)`

Transforms input text into its binary representation with configurable bits per character (8, 16, or 32).

**Constructor Parameters:**

| Parameter | Type | Description |
|---|---|---|
| `bits_per_char` | `BinaryConverter.BitsPerChar` | Number of bits to use for each character (8, 16, or 32). Default is 16 bits. Defaults to `BitsPerChar.BITS_16`. |
| `word_selection_strategy` | `Optional[WordSelectionStrategy]` | Strategy for selecting which words to convert. If None, all words will be converted. Defaults to `None`. |

**Methods:**

#### `convert_word_async(word: str) → str`

Convert a single word into the target format supported by the converter.

| Parameter | Type | Description |
|---|---|---|
| `word` | `str` | The word to be converted. |

**Returns:**

- `str` — The converted word.

#### `join_words(words: list[str]) → str`

Join the converted words with the binary representation of a space.

| Parameter | Type | Description |
|---|---|---|
| `words` | `list[str]` | The list of converted words. |

**Returns:**

- `str` — The final joined string with spaces in binary format.

#### `validate_input(prompt: str) → None`

Check if ``bits_per_char`` is sufficient for the characters in the prompt.

| Parameter | Type | Description |
|---|---|---|
| `prompt` | `str` | The input text prompt to validate. |

**Raises:**

- `ValueError` — If ``bits_per_char`` is too small to represent any character in the prompt.

## `class BrailleConverter(PromptConverter)`

Converts text into Braille Unicode representation.

This converter transforms standard text into Braille patterns using Unicode
Braille characters (U+2800 to U+28FF). It supports lowercase and uppercase
letters, numbers, common punctuation, and spaces. Uppercase letters are
prefixed with the Braille capitalization indicator.

The Braille mapping is based on the implementation from Garak:
https://github.com/NVIDIA/garak/blob/main/garak/probes/encoding.py

Note: This converter is useful for testing how AI systems handle Braille-encoded
text, which can be used to obfuscate potentially harmful content.

**Methods:**

#### convert_async

```python
convert_async(prompt: str, input_type: PromptDataType = 'text') → ConverterResult
```

Convert the given text into Braille Unicode representation.

| Parameter | Type | Description |
|---|---|---|
| `prompt` | `str` | The text to be converted to Braille. |
| `input_type` | `PromptDataType` | Type of input data. Defaults to "text". Defaults to `'text'`. |

**Returns:**

- `ConverterResult` — The text converted to Braille Unicode characters.

**Raises:**

- `ValueError` — If the input type is not supported (only "text" is supported).

## `class CaesarConverter(PromptConverter)`

Encodes text using the Caesar cipher with a specified offset.

Using ``offset=1``, 'Hello 123' would encode to 'Ifmmp 234', as each character would shift by 1.
Shifts for digits 0-9 only work if the offset is less than 10, if the offset is equal to or greater than 10,
any numeric values will not be shifted.

**Constructor Parameters:**

| Parameter | Type | Description |
|---|---|---|
| `caesar_offset` | `int` | Offset for caesar cipher, range 0 to 25 (inclusive). Can also be negative for shifting backwards. |
| `append_description` | `bool` | If True, appends plaintext "expert" text to the prompt. This includes instructions to only communicate using the cipher, a description of the cipher, and an example encoded using the cipher. Defaults to `False`. |

**Methods:**

#### convert_async

```python
convert_async(prompt: str, input_type: PromptDataType = 'text') → ConverterResult
```

Convert the given prompt using the Caesar cipher.

| Parameter | Type | Description |
|---|---|---|
| `prompt` | `str` | The input prompt to be converted. |
| `input_type` | `PromptDataType` | The type of the input prompt. Must be "text". Defaults to `'text'`. |

**Returns:**

- `ConverterResult` — The result containing the converted prompt and its type.

**Raises:**

- `ValueError` — If the input type is not supported.

## `class CharSwapConverter(WordLevelConverter)`

Applies character swapping to words in the prompt to test adversarial textual robustness.

**Constructor Parameters:**

| Parameter | Type | Description |
|---|---|---|
| `max_iterations` | `int` | Number of times to generate perturbed prompts. The higher the number the higher the chance that words are different from the original prompt. Defaults to `10`. |
| `word_selection_strategy` | `Optional[WordSelectionStrategy]` | Strategy for selecting which words to convert. If None, defaults to WordProportionSelectionStrategy(proportion=0.2). Defaults to `None`. |

**Methods:**

#### `convert_word_async(word: str) → str`

Convert a single word into the target format supported by the converter.

| Parameter | Type | Description |
|---|---|---|
| `word` | `str` | The word to be converted. |

**Returns:**

- `str` — The converted word.

## `class CharacterSpaceConverter(PromptConverter)`

Spaces out the input prompt and removes specified punctuations.

For more information on the bypass strategy, refer to:
https://www.robustintelligence.com/blog-posts/bypassing-metas-llama-classifier-a-simple-jailbreak

**Methods:**

#### convert_async

```python
convert_async(prompt: str, input_type: PromptDataType = 'text') → ConverterResult
```

Convert the given prompt by removing punctuation and spacing out characters.

| Parameter | Type | Description |
|---|---|---|
| `prompt` | `str` | The input text prompt to be converted. |
| `input_type` | `PromptDataType` | The type of input data. Defaults to `'text'`. |

**Returns:**

- `ConverterResult` — The result containing the converted text.

**Raises:**

- `ValueError` — If the input type is not supported.

## `class CodeChameleonConverter(PromptConverter)`

Code Chameleon Converter based on https://arxiv.org/abs/2402.16717 by Lv, Huijie, et al.

**Constructor Parameters:**

| Parameter | Type | Description |
|---|---|---|
| `encrypt_type` | `str` | Must be one of "custom", "reverse", "binary_tree", "odd_even" or "length". |
| `encrypt_function` | `Callable` | User provided encryption function. Only used if `encrypt_mode` is "custom". Used to encode user prompt. Defaults to `None`. |
| `decrypt_function` | `Callable or list` | User provided encryption function. Only used if `encrypt_mode` is "custom". Used as part of markdown code block instructions in system prompt. If list is provided, strings will be treated as single statements for imports or comments. Functions will take the source code of the function. Defaults to `None`. |

**Methods:**

#### convert_async

```python
convert_async(prompt: str, input_type: PromptDataType = 'text') → ConverterResult
```

Convert the given prompt by applying the specified encryption function.

| Parameter | Type | Description |
|---|---|---|
| `prompt` | `str` | The input prompt to be converted. |
| `input_type` | `PromptDataType` | The type of input data. Defaults to `'text'`. |

**Returns:**

- `ConverterResult` — The result containing the converted prompt.

**Raises:**

- `ValueError` — If the input type is not supported.

## `class ColloquialWordswapConverter(PromptConverter)`

Converts text by replacing words with regional colloquial alternatives.

Supports loading substitutions from YAML files (e.g., Singaporean, Filipino, Indian)
or accepting a custom substitution dictionary. Defaults to Singaporean substitutions.

**Constructor Parameters:**

| Parameter | Type | Description |
|---|---|---|
| `*args` | `bool` | Deprecated positional argument for deterministic. Use deterministic=... instead. Defaults to `()`. |
| `deterministic` | `bool` | If True, use the first substitution for each wordswap. If False, randomly choose a substitution for each wordswap. Defaults to False. Defaults to `False`. |
| `custom_substitutions` | `Optional[dict[str, list[str]]]` | A dictionary of custom substitutions to override the defaults. Defaults to None. Defaults to `None`. |
| `wordswap_path` | `Optional[str]` | Path to a YAML file containing word substitutions. Can be a filename within the built-in colloquial_wordswaps directory (e.g., "filipino.yaml") or an absolute path to a custom YAML file. Defaults to None (uses singaporean.yaml). Defaults to `None`. |

**Methods:**

#### convert_async

```python
convert_async(prompt: str, input_type: PromptDataType = 'text') → ConverterResult
```

Convert the given prompt by replacing words with regional colloquial alternatives.

| Parameter | Type | Description |
|---|---|---|
| `prompt` | `str` | The input text prompt to be converted. |
| `input_type` | `PromptDataType` | The type of the input prompt. Defaults to "text". Defaults to `'text'`. |

**Returns:**

- `ConverterResult` — The result containing the converted prompt.

**Raises:**

- `ValueError` — If the input type is not supported.

## `class ConverterResult`

The result of a prompt conversion, containing the converted output and its type.

## `class DenylistConverter(LLMGenericTextConverter)`

Replaces forbidden words or phrases in a prompt with synonyms using an LLM.

An existing ``PromptChatTarget`` is used to perform the conversion (like Azure OpenAI).

**Constructor Parameters:**

| Parameter | Type | Description |
|---|---|---|
| `converter_target` | `PromptChatTarget` | The target for the prompt conversion. Can be omitted if a default has been configured via PyRIT initialization. Defaults to `REQUIRED_VALUE`. |
| `system_prompt_template` | `Optional[SeedPrompt]` | The system prompt template to use for the conversion. If not provided, a default template will be used. Defaults to `None`. |
| `denylist` | `list[str]` | A list of words or phrases that should be replaced in the prompt. Defaults to `None`. |

**Methods:**

#### convert_async

```python
convert_async(prompt: str, input_type: PromptDataType = 'text') → ConverterResult
```

Convert the given prompt by removing any words or phrases that are in the denylist,
replacing them with synonymous words.

| Parameter | Type | Description |
|---|---|---|
| `prompt` | `str` | The prompt to be converted. |
| `input_type` | `PromptDataType` | The type of input data. Defaults to `'text'`. |

**Returns:**

- `ConverterResult` — The result containing the modified prompt.

## `class DiacriticConverter(PromptConverter)`

Applies diacritics to specified characters in a string.

**Constructor Parameters:**

| Parameter | Type | Description |
|---|---|---|
| `target_chars` | `str` | Characters to apply the diacritic to. Defaults to "aeiou". Defaults to `'aeiou'`. |
| `accent` | `str` | Type of diacritic to apply (default is 'acute').  Available options are:     - `acute`: ́     - `grave`: ̀     - `tilde`: ̃     - `umlaut`: ̈ Defaults to `'acute'`. |

**Methods:**

#### convert_async

```python
convert_async(prompt: str, input_type: PromptDataType = 'text') → ConverterResult
```

Convert the given prompt by applying diacritics to specified characters.

| Parameter | Type | Description |
|---|---|---|
| `prompt` | `str` | The text prompt to be converted. |
| `input_type` | `PromptDataType` | The type of input data. Defaults to `'text'`. |

**Returns:**

- `ConverterResult` — The result containing the modified text.

**Raises:**

- `ValueError` — If the input type is not supported.

## `class EcojiConverter(PromptConverter)`

Converter that encodes text using Ecoji encoding.

Ecoji is an encoding scheme that represents binary data using emojis.
See https://ecoji.io/ for more details.

**Methods:**

#### convert_async

```python
convert_async(prompt: str, input_type: PromptDataType = 'text') → ConverterResult
```

Convert the given prompt to Ecoji encoding.

| Parameter | Type | Description |
|---|---|---|
| `prompt` | `str` | The text to encode. |
| `input_type` | `PromptDataType` | The type of input. Defaults to "text". Defaults to `'text'`. |

**Returns:**

- `ConverterResult` — The result containing the Ecoji-encoded text.

**Raises:**

- `ValueError` — If the input type is not supported.

## `class EmojiConverter(WordLevelConverter)`

Converts English text to randomly chosen circle or square character emojis.

Inspired by https://github.com/BASI-LABS/parseltongue/blob/main/src/utils.ts

**Methods:**

#### `convert_word_async(word: str) → str`

Convert a single word into the target format supported by the converter.

| Parameter | Type | Description |
|---|---|---|
| `word` | `str` | The word to be converted. |

**Returns:**

- `str` — The converted word.

## `class FirstLetterConverter(WordLevelConverter)`

Replaces each word of the prompt with its first letter (or digit).
Whitespace and words that do not contain any letter or digit are ignored.

**Constructor Parameters:**

| Parameter | Type | Description |
|---|---|---|
| `letter_separator` | `str` | The string used to join the first letters. Defaults to `' '`. |
| `word_selection_strategy` | `Optional[WordSelectionStrategy]` | Strategy for selecting which words to convert. If None, all words will be converted. Defaults to `None`. |

**Methods:**

#### `convert_word_async(word: str) → str`

Convert a single word into the target format supported by the converter.

| Parameter | Type | Description |
|---|---|---|
| `word` | `str` | The word to be converted. |

**Returns:**

- `str` — The converted word.

#### `join_words(words: list[str]) → str`

Join the converted words using the specified letter separator.

| Parameter | Type | Description |
|---|---|---|
| `words` | `list[str]` | The list of converted words. |

**Returns:**

- `str` — The joined string of converted words.

## `class FlipConverter(PromptConverter)`

Flips the input text prompt. For example, "hello me" would be converted to "em olleh".

**Methods:**

#### convert_async

```python
convert_async(prompt: str, input_type: PromptDataType = 'text') → ConverterResult
```

Convert the given prompt by reversing the text.

| Parameter | Type | Description |
|---|---|---|
| `prompt` | `str` | The prompt to be converted. |
| `input_type` | `PromptDataType` | Type of data. Defaults to `'text'`. |

**Returns:**

- `ConverterResult` — The converted text representation of the original prompt with characters reversed.

**Raises:**

- `ValueError` — If the input type is not supported.

## `class HumanInTheLoopConverter(PromptConverter)`

Allows review of each prompt sent to a target before sending it.

Users can choose to send the prompt as is, modify the prompt,
or run the prompt through one of the passed-in converters before sending it.

.. deprecated::
    This converter is deprecated and will be removed in v0.13.0.
    Use the React-based GUI (CoPyRIT) instead.

**Constructor Parameters:**

| Parameter | Type | Description |
|---|---|---|
| `converters` | `(List[PromptConverter], Optional)` | List of possible converters to run input through. Defaults to `None`. |

**Methods:**

#### convert_async

```python
convert_async(prompt: str, input_type: PromptDataType = 'text') → ConverterResult
```

Convert the given prompt by allowing user interaction before sending it to a target.

| Parameter | Type | Description |
|---|---|---|
| `prompt` | `str` | The prompt to be converted. |
| `input_type` | `PromptDataType` | The type of input data. Defaults to `'text'`. |

**Returns:**

- `ConverterResult` — The result containing the modified prompt.

**Raises:**

- `ValueError` — If no converters are provided and the user chooses to run a converter.

## `class ImageCompressionConverter(PromptConverter)`

Compresses images to reduce file size while preserving visual quality.

This converter supports multiple compression strategies across JPEG, PNG, and WEBP formats,
each with format-specific optimization settings. It can maintain the original image format
or convert between formats as needed.

When converting images with transparency (alpha channel) to JPEG format, the converter
automatically composites the transparent areas onto a solid background color.

Supported input types:
File paths to any image that PIL can open (or URLs pointing to such images):
https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html#fully-supported-formats

**Constructor Parameters:**

| Parameter | Type | Description |
|---|---|---|
| `output_format` | `str` | Output image format. If None, keeps original format (if supported). Defaults to `None`. |
| `quality` | `int` | General quality setting for JPEG and WEBP formats (0-100).  For JPEG format, it represents the image quality, on a scale from 0 (worst) to 95 (best).  For WEBP format, the value ranges from 0 to 100; for lossy compression: 0-smallest file size and 100-largest; for ``lossless``: 0-fastest/less efficient, and 100 gives the best compression. Defaults to `None`. |
| `optimize` | `bool` | Whether to optimize the image during compression.   For JPEG: makes the encoder perform an extra pass over the image to select optimal settings.  For PNG: instructs the PNG writer to make the output file as small as possible. Defaults to `None`. |
| `progressive` | `bool` | Whether to save JPEG images as progressive. Defaults to `None`. |
| `compress_level` | `int` | ZLIB compression level (0-9): 1=fastest, 9=best, 0=none. Ignored if ``optimize`` is True (then it is forced to 9). Defaults to `None`. |
| `lossless` | `bool` | Whether to use lossless compression for WEBP format. Defaults to `None`. |
| `method` | `int` | Quality/speed trade-off for WEBP format (0=fast, 6=slower-better). Defaults to `None`. |
| `background_color` | `tuple[int, int, int]` | RGB color tuple for background when converting transparent images to JPEG. Defaults to black. Defaults to `(0, 0, 0)`. |
| `min_compression_threshold` | `int` | Minimum file size threshold for compression. Defaults to 1024 bytes. Defaults to `1024`. |
| `fallback_to_original` | `bool` | Fallback to original if compression increases file size. Defaults to True. Defaults to `True`. |

**Methods:**

#### convert_async

```python
convert_async(prompt: str, input_type: PromptDataType = 'image_path') → ConverterResult
```

Convert the given prompt (image) by compressing it.

| Parameter | Type | Description |
|---|---|---|
| `prompt` | `str` | The image file path or URL pointing to the image to be compressed. |
| `input_type` | `PromptDataType` | The type of input data. Defaults to `'image_path'`. |

**Returns:**

- `ConverterResult` — The result containing path to the compressed image.

**Raises:**

- `ValueError` — If the input type is not supported.

## `class IndexSelectionStrategy(TextSelectionStrategy)`

Selects text based on absolute character indices.

**Constructor Parameters:**

| Parameter | Type | Description |
|---|---|---|
| `start` | `int` | The starting character index (inclusive). Defaults to 0. Defaults to `0`. |
| `end` | `Optional[int]` | The ending character index (exclusive). If None, selects to end of text. Defaults to `None`. |

**Methods:**

#### `select_range(text: str) → tuple[int, int]`

Select a range based on absolute character indices.

| Parameter | Type | Description |
|---|---|---|
| `text` | `str` | The input text to select from. |

**Returns:**

- `tuple[int, int]` — tuple[int, int]: A tuple of (start_index, end_index).

## `class InsertPunctuationConverter(PromptConverter)`

Inserts punctuation into a prompt to test robustness.

Punctuation insertion: inserting single punctuations in `string.punctuation`.
Words in a prompt: a word does not contain any punctuation and space.
"a1b2c3" is a word; "a1 2" are 2 words; "a1,b,3" are 3 words.

**Constructor Parameters:**

| Parameter | Type | Description |
|---|---|---|
| `word_swap_ratio` | `float` | Percentage of words to perturb. Defaults to 0.2. Defaults to `0.2`. |
| `between_words` | `bool` | If True, insert punctuation only between words. If False, insert punctuation within words. Defaults to True. Defaults to `True`. |

**Methods:**

#### convert_async

```python
convert_async(prompt: str, input_type: PromptDataType = 'text', punctuation_list: Optional[list[str]] = None) → ConverterResult
```

Convert the given prompt by inserting punctuation.

| Parameter | Type | Description |
|---|---|---|
| `prompt` | `str` | The text to convert. |
| `input_type` | `PromptDataType` | The type of input data. Defaults to `'text'`. |
| `punctuation_list` | `Optional[List[str]]` | List of punctuations to use for insertion. Defaults to `None`. |

**Returns:**

- `ConverterResult` — The result containing an iteration of modified prompts.

**Raises:**

- `ValueError` — If the input type is not supported.

## `class JsonStringConverter(PromptConverter)`

Converts a string to a JSON-safe format using json.dumps().

This converter is useful when a string needs to be embedded within a JSON payload,
such as when sending prompts to HTTP targets that expect JSON-formatted requests.
The converter properly escapes special characters like quotes, newlines, backslashes,
and unicode characters.

The output is the escaped string content without the surrounding quotes that
json.dumps() adds, making it ready to be inserted into a JSON string field.

**Methods:**

#### convert_async

```python
convert_async(prompt: str, input_type: PromptDataType = 'text') → ConverterResult
```

Convert the given prompt to a JSON-safe string.

| Parameter | Type | Description |
|---|---|---|
| `prompt` | `str` | The prompt to be converted. |
| `input_type` | `PromptDataType` | The type of input data. Defaults to `'text'`. |

**Returns:**

- `ConverterResult` — The result containing the JSON-escaped string.

**Raises:**

- `ValueError` — If the input type is not supported.

## `class KeywordSelectionStrategy(TextSelectionStrategy)`

Selects text around a keyword with optional context.

**Constructor Parameters:**

| Parameter | Type | Description |
|---|---|---|
| `keyword` | `str` | The keyword to search for. |
| `context_before` | `int` | Number of characters to include before the keyword. Defaults to 0. Defaults to `0`. |
| `context_after` | `int` | Number of characters to include after the keyword. Defaults to 0. Defaults to `0`. |
| `case_sensitive` | `bool` | Whether the keyword search is case-sensitive. Defaults to True. Defaults to `True`. |

**Methods:**

#### `select_range(text: str) → tuple[int, int]`

Select the range around the first occurrence of the keyword.

| Parameter | Type | Description |
|---|---|---|
| `text` | `str` | The input text to select from. |

**Returns:**

- `tuple[int, int]` — tuple[int, int]: A tuple of (start_index, end_index) including context,
or (0, 0) if keyword not found.

## `class LLMGenericTextConverter(PromptConverter)`

Represents a generic LLM converter that expects text to be transformed (e.g. no JSON parsing or format).

**Constructor Parameters:**

| Parameter | Type | Description |
|---|---|---|
| `converter_target` | `PromptChatTarget` | The endpoint that converts the prompt. Can be omitted if a default has been configured via PyRIT initialization. Defaults to `REQUIRED_VALUE`. |
| `system_prompt_template` | `(SeedPrompt, Optional)` | The prompt template to set as the system prompt. Defaults to `None`. |
| `user_prompt_template_with_objective` | `(SeedPrompt, Optional)` | The prompt template to set as the user prompt. expects Defaults to `None`. |
| `kwargs` | `Any` | Additional parameters for the prompt template. Defaults to `{}`. |

**Methods:**

#### convert_async

```python
convert_async(prompt: str, input_type: PromptDataType = 'text') → ConverterResult
```

Convert the given prompt using an LLM via the specified converter target.

| Parameter | Type | Description |
|---|---|---|
| `prompt` | `str` | The prompt to be converted. |
| `input_type` | `PromptDataType` | The type of input data. Defaults to `'text'`. |

**Returns:**

- `ConverterResult` — The result containing the converted output and its type.

**Raises:**

- `ValueError` — If the input type is not supported.

## `class LeetspeakConverter(WordLevelConverter)`

Converts a string to a leetspeak version.

**Constructor Parameters:**

| Parameter | Type | Description |
|---|---|---|
| `deterministic` | `bool` | If True, use the first substitution for each character. If False, randomly choose a substitution for each character. Defaults to `True`. |
| `custom_substitutions` | `Optional[dict]` | A dictionary of custom substitutions to override the defaults. Defaults to `None`. |
| `word_selection_strategy` | `Optional[WordSelectionStrategy]` | Strategy for selecting which words to convert. If None, all words will be converted. Defaults to `None`. |

**Methods:**

#### `convert_word_async(word: str) → str`

Convert a single word into the target format supported by the converter.

| Parameter | Type | Description |
|---|---|---|
| `word` | `str` | The word to be converted. |

**Returns:**

- `str` — The converted word.

## `class MaliciousQuestionGeneratorConverter(LLMGenericTextConverter)`

Generates malicious questions using an LLM.

An existing ``PromptChatTarget`` is used to perform the conversion (like Azure OpenAI).

**Constructor Parameters:**

| Parameter | Type | Description |
|---|---|---|
| `converter_target` | `PromptChatTarget` | The endpoint that converts the prompt. Can be omitted if a default has been configured via PyRIT initialization. Defaults to `REQUIRED_VALUE`. |
| `prompt_template` | `SeedPrompt` | The seed prompt template to use. Defaults to `None`. |

**Methods:**

#### convert_async

```python
convert_async(prompt: str, input_type: PromptDataType = 'text') → ConverterResult
```

Convert the input prompt into malicious questions.

| Parameter | Type | Description |
|---|---|---|
| `prompt` | `str` | The input prompt to convert. |
| `input_type` | `PromptDataType` | The type of the input prompt. Must be "text". Defaults to `'text'`. |

**Returns:**

- `ConverterResult` — The result of the conversion.

#### `input_supported(input_type: PromptDataType) → bool`

Check if the input type is supported.

| Parameter | Type | Description |
|---|---|---|
| `input_type` | `PromptDataType` | The type of the input prompt. |

**Returns:**

- `bool` — True if the input type is "text", False otherwise.

#### `output_supported(output_type: PromptDataType) → bool`

Check if the output type is supported.

| Parameter | Type | Description |
|---|---|---|
| `output_type` | `PromptDataType` | The desired type of the output prompt. |

**Returns:**

- `bool` — True if the output type is "text", False otherwise.

## `class MathObfuscationConverter(PromptConverter)`

Convert text into character-level algebraic identities.

This converter encodes each character of the input text into an equation of
the form `X = nX - (n - 1)X`, where `n` is a randomly chosen integer greater
than or equal to 2. This creates a deterministic, reversible obfuscation of
the original input.

The transformation follows these rules:

* Each non-space character becomes one algebraic line.
* Space characters are represented as blank output lines.
* Newline characters are preserved as blank output lines.

An inline hint is added after the first equation, and a suffix instruction
is appended to prompt the model to decode the content.

**Constructor Parameters:**

| Parameter | Type | Description |
|---|---|---|
| `min_n` | `int` |  Minimum integer value used for `n` in `X = nX - (n - 1)X`. Must be greater than or equal to 2. Defaults to `2`. |
| `max_n` | `int` |  Maximum integer value used for `n`. Must be greater than or equal to `min_n`. Defaults to `9`. |
| `hint` | `Optional[str]` |  Inline hint appended to the first equation line. If None, uses the default hint explaining the variable encoding. Set to empty string "" to disable hint entirely. Defaults to `None`. |
| `suffix` | `Optional[str]` |  Custom suffix to append after the obfuscated text. If None, uses the default suffix prompting the model to decode. Set to empty string "" to disable suffix entirely. Defaults to `None`. |
| `rng` | `Optional[random.Random]` |  Optional random number generator instance used to produce reproducible obfuscation results. If omitted, a new instance of `random.Random()` is created. Defaults to `None`. |

**Methods:**

#### convert_async

```python
convert_async(prompt: str, input_type: PromptDataType = 'text') → ConverterResult
```

Convert text into algebraic obfuscated form.

Each non-space character in the input string is transformed into a
corresponding algebraic identity. Space characters are represented as
blank output lines, preserving word boundaries. Newline characters are
preserved as block breaks in the output.

| Parameter | Type | Description |
|---|---|---|
| `prompt` | `str` |  Input text to be transformed. |
| `input_type` | `PromptDataType` |  Expected to be `"text"`. Other types are not supported. Defaults to `'text'`. |

**Returns:**

- `ConverterResult` —
An instance containing the obfuscated text and output format.

**Raises:**

- `ValueError` — If `input_type` is not `"text"`.

## `class MathPromptConverter(LLMGenericTextConverter)`

Converts natural language instructions into symbolic mathematics problems using an LLM.

An existing ``PromptChatTarget`` is used to perform the conversion (like Azure OpenAI).

**Constructor Parameters:**

| Parameter | Type | Description |
|---|---|---|
| `converter_target` | `PromptChatTarget` | The endpoint that converts the prompt. Can be omitted if a default has been configured via PyRIT initialization. Defaults to `REQUIRED_VALUE`. |
| `prompt_template` | `SeedPrompt` | The seed prompt template to use. Defaults to `None`. |

**Methods:**

#### convert_async

```python
convert_async(prompt: str, input_type: PromptDataType = 'text') → ConverterResult
```

Convert the given prompt into a mathematical problem format.

| Parameter | Type | Description |
|---|---|---|
| `prompt` | `str` | The prompt to be converted. |
| `input_type` | `PromptDataType` | The type of input data. Defaults to `'text'`. |

**Returns:**

- `ConverterResult` — The result containing the mathematical representation and real-world example.

## `class MorseConverter(PromptConverter)`

Encodes prompts using morse code.

Uses '-' and '.' characters, with ' ' to separate characters and '/' to separate words.
Invalid or unsupported characters are replaced with an error sequence '........'.

**Constructor Parameters:**

| Parameter | Type | Description |
|---|---|---|
| `append_description` | `bool` | Append plaintext "expert" text to the prompt. Includes instructions to only communicate using the cipher, a description of the cipher, and an example encoded using cipher. Defaults to `False`. |

**Methods:**

#### convert_async

```python
convert_async(prompt: str, input_type: PromptDataType = 'text') → ConverterResult
```

Convert the given prompt to morse code.

| Parameter | Type | Description |
|---|---|---|
| `prompt` | `str` | The prompt to be converted. |
| `input_type` | `PromptDataType` | Type of input data. Defaults to "text". Defaults to `'text'`. |

**Returns:**

- `ConverterResult` — The result containing the morse code representation of the prompt.

**Raises:**

- `ValueError` — If the input type is not supported.

## `class NatoConverter(PromptConverter)`

Converts text into NATO phonetic alphabet representation.

This converter transforms standard text into NATO phonetic alphabet format,
where each letter is replaced with its corresponding NATO phonetic code word
(e.g., "A" becomes "Alfa", "B" becomes "Bravo"). Only alphabetic characters
are converted; non-alphabetic characters are ignored.

The NATO phonetic alphabet is the most widely used spelling alphabet, designed
to improve clarity of voice communication. This converter can be used to test
how AI systems handle phonetically encoded text, which can be used to obfuscate
potentially harmful content.

Reference: https://en.wikipedia.org/wiki/NATO_phonetic_alphabet

**Methods:**

#### convert_async

```python
convert_async(prompt: str, input_type: PromptDataType = 'text') → ConverterResult
```

Convert the given text into NATO phonetic alphabet representation.

| Parameter | Type | Description |
|---|---|---|
| `prompt` | `str` | The text to be converted to NATO phonetic alphabet. |
| `input_type` | `PromptDataType` | Type of input data. Defaults to "text". Defaults to `'text'`. |

**Returns:**

- `ConverterResult` — The text converted to NATO phonetic alphabet format.

**Raises:**

- `ValueError` — If the input type is not supported (only "text" is supported).

## `class NegationTrapConverter(PromptConverter)`

Converts a prompt into a negation-based logical trap. This technique
exploits LLM reasoning patterns by asking the model to confirm or deny
a wrong answer, potentially causing it to reveal the correct one.

This technique was discovered during CTF red teaming exercises where
targets would leak information when asked to evaluate incorrect statements.

**Constructor Parameters:**

| Parameter | Type | Description |
|---|---|---|
| `wrong_value` | `str` | A deliberately wrong value to use in the trap. The target         may reveal the correct value when correcting this. Defaults to `'incorrect_guess'`. |
| `trap_template` | `str | None` | A custom template string. Must include {prompt} and {wrong_value}           placeholders. If None, uses the default denial template. Defaults to `None`. |

**Methods:**

#### convert_async

```python
convert_async(prompt: str, input_type: PromptDataType = 'text') → ConverterResult
```

Convert the prompt into a negation trap.

This technique works by presenting an obviously wrong answer and asking
the target to correct it, which may cause it to reveal protected information.

| Parameter | Type | Description |
|---|---|---|
| `prompt` | `str` | The prompt to be converted. |
| `input_type` | `PromptDataType` | The type of the input prompt. Defaults to `'text'`. |

**Returns:**

- `ConverterResult` — The prompt converted to a negation trap.

**Raises:**

- `ValueError` — If the input type is not supported.

#### `input_supported(input_type: PromptDataType) → bool`

Check if the input type is supported.

| Parameter | Type | Description |
|---|---|---|
| `input_type` | `PromptDataType` | The type of the input prompt. |

**Returns:**

- `bool` — True if the input type is supported, False otherwise.

#### `output_supported(output_type: PromptDataType) → bool`

Check if the output type is supported.

| Parameter | Type | Description |
|---|---|---|
| `output_type` | `PromptDataType` | The desired type of the output prompt. |

**Returns:**

- `bool` — True if the output type is supported, False otherwise.

## `class NoiseConverter(LLMGenericTextConverter)`

Injects noise errors into a conversation using an LLM.

An existing ``PromptChatTarget`` is used to perform the conversion (like Azure OpenAI).

**Constructor Parameters:**

| Parameter | Type | Description |
|---|---|---|
| `converter_target` | `PromptChatTarget` | The endpoint that converts the prompt. Can be omitted if a default has been configured via PyRIT initialization. Defaults to `REQUIRED_VALUE`. |
| `noise` | `str` | The noise to inject. Grammar error, delete random letter, insert random space, etc. Defaults to `None`. |
| `number_errors` | `int` | The number of errors to inject. Defaults to `5`. |
| `prompt_template` | `(SeedPrompt, Optional)` | The prompt template for the conversion. Defaults to `None`. |

## `class PDFConverter(PromptConverter)`

Converts a text prompt into a PDF file.

**Constructor Parameters:**

| Parameter | Type | Description |
|---|---|---|
| `prompt_template` | `Optional[SeedPrompt]` | A ``SeedPrompt`` object representing a template. Defaults to `None`. |
| `font_type` | `str` | Font type for the PDF. Defaults to "Helvetica". Defaults to `'Helvetica'`. |
| `font_size` | `int` | Font size for the PDF. Defaults to 12. Defaults to `12`. |
| `font_color` | `tuple` | Font color for the PDF in RGB format. Defaults to (255, 255, 255). Defaults to `(255, 255, 255)`. |
| `page_width` | `int` | Width of the PDF page in mm. Defaults to 210 (A4 width). Defaults to `210`. |
| `page_height` | `int` | Height of the PDF page in mm. Defaults to 297 (A4 height). Defaults to `297`. |
| `column_width` | `int` | Width of each column in the PDF. Defaults to 0 (full page width). Defaults to `0`. |
| `row_height` | `int` | Height of each row in the PDF. Defaults to 10. Defaults to `10`. |
| `existing_pdf` | `Optional[Path]` | Path to an existing PDF file. Defaults to None. Defaults to `None`. |
| `injection_items` | `Optional[List[Dict]]` | A list of injection items for modifying an existing PDF. Defaults to `None`. |

**Methods:**

#### convert_async

```python
convert_async(prompt: str, input_type: PromptDataType = 'text') → ConverterResult
```

Convert the given prompt into a PDF.

If a template is provided, it injects the prompt into the template, otherwise, it generates
a simple PDF with the prompt as the content. Further it can modify existing PDFs.

| Parameter | Type | Description |
|---|---|---|
| `prompt` | `str` | The prompt to be embedded in the PDF. |
| `input_type` | `PromptDataType` | The type of input data. Defaults to `'text'`. |

**Returns:**

- `ConverterResult` — The result containing the full file path to the generated PDF.

**Raises:**

- `ValueError` — If the input type is not supported.

## `class PersuasionConverter(PromptConverter)`

Rephrases prompts using a variety of persuasion techniques.

Based on https://arxiv.org/abs/2401.06373 by Zeng et al.

**Constructor Parameters:**

| Parameter | Type | Description |
|---|---|---|
| `converter_target` | `PromptChatTarget` | The chat target used to perform rewriting on user prompts. Can be omitted if a default has been configured via PyRIT initialization. Defaults to `REQUIRED_VALUE`. |
| `persuasion_technique` | `str` | Persuasion technique to be used by the converter, determines the system prompt to be used to generate new prompts. Must be one of "authority_endorsement", "evidence_based", "expert_endorsement", "logical_appeal", "misrepresentation". |

**Methods:**

#### convert_async

```python
convert_async(prompt: str, input_type: PromptDataType = 'text') → ConverterResult
```

Convert the given prompt using the persuasion technique specified during initialization.

| Parameter | Type | Description |
|---|---|---|
| `prompt` | `str` | The input prompt to be converted. |
| `input_type` | `PromptDataType` | The type of input data. Defaults to `'text'`. |

**Returns:**

- `ConverterResult` — The result containing the converted prompt text.

**Raises:**

- `ValueError` — If the input type is not supported.

#### `send_persuasion_prompt_async(request: Message) → str`

Send the prompt to the converter target and process the response.

| Parameter | Type | Description |
|---|---|---|
| `request` | `Message` | The message containing the prompt to be converted. |

**Returns:**

- `str` — The converted prompt text extracted from the response.

**Raises:**

- `InvalidJsonException` — If the response is not valid JSON or missing expected keys.

## `class PositionSelectionStrategy(TextSelectionStrategy)`

Selects text based on proportional start and end positions.

**Constructor Parameters:**

| Parameter | Type | Description |
|---|---|---|
| `start_proportion` | `float` | The starting position as a proportion (0.0 to 1.0). |
| `end_proportion` | `float` | The ending position as a proportion (0.0 to 1.0). |

**Methods:**

#### `select_range(text: str) → tuple[int, int]`

Select a range based on the relative position in the text.

| Parameter | Type | Description |
|---|---|---|
| `text` | `str` | The input text to select from. |

**Returns:**

- `tuple[int, int]` — tuple[int, int]: A tuple of (start_index, end_index).

## `class PromptConverter(Identifiable)`

Base class for converters that transform prompts into a different representation or format.

Concrete subclasses must declare their supported input and output modalities using class attributes:
- SUPPORTED_INPUT_TYPES: tuple of PromptDataType values that the converter accepts
- SUPPORTED_OUTPUT_TYPES: tuple of PromptDataType values that the converter produces

These attributes are enforced at class definition time for all non-abstract subclasses.

**Methods:**

#### convert_async

```python
convert_async(prompt: str, input_type: PromptDataType = 'text') → ConverterResult
```

Convert the given prompt into the target format supported by the converter.

| Parameter | Type | Description |
|---|---|---|
| `prompt` | `str` | The prompt to be converted. |
| `input_type` | `PromptDataType` | The type of input data. Defaults to `'text'`. |

**Returns:**

- `ConverterResult` — The result containing the converted output and its type.

#### convert_tokens_async

```python
convert_tokens_async(prompt: str, input_type: PromptDataType = 'text', start_token: str = '⟪', end_token: str = '⟫') → ConverterResult
```

Convert substrings within a prompt that are enclosed by specified start and end tokens. If there are no tokens
present, the entire prompt is converted.

| Parameter | Type | Description |
|---|---|---|
| `prompt` | `str` | The input prompt containing text to be converted. |
| `input_type` | `str` | The type of input data. Defaults to "text". Defaults to `'text'`. |
| `start_token` | `str` | The token indicating the start of a substring to be converted. Defaults to "⟪" which is relatively distinct. Defaults to `'⟪'`. |
| `end_token` | `str` | The token indicating the end of a substring to be converted. Defaults to "⟫" which is relatively distinct. Defaults to `'⟫'`. |

**Returns:**

- `ConverterResult` — The prompt with specified substrings converted.

**Raises:**

- `ValueError` — If the input is inconsistent.

#### `input_supported(input_type: PromptDataType) → bool`

Check if the input type is supported by the converter.

| Parameter | Type | Description |
|---|---|---|
| `input_type` | `PromptDataType` | The input type to check. |

**Returns:**

- `bool` — True if the input type is supported, False otherwise.

#### `output_supported(output_type: PromptDataType) → bool`

Check if the output type is supported by the converter.

| Parameter | Type | Description |
|---|---|---|
| `output_type` | `PromptDataType` | The output type to check. |

**Returns:**

- `bool` — True if the output type is supported, False otherwise.

## `class ProportionSelectionStrategy(TextSelectionStrategy)`

Selects a proportion of text anchored to a specific position (start, end, middle, or random).

**Constructor Parameters:**

| Parameter | Type | Description |
|---|---|---|
| `proportion` | `float` | The proportion of text to select (0.0 to 1.0). |
| `anchor` | `str` | Where to anchor the selection. Valid values: - 'start': Select from the beginning - 'end': Select from the end - 'middle': Select from the middle - 'random': Select from a random position Defaults to `'start'`. |
| `seed` | `Optional[int]` | Random seed for reproducible random selections. Defaults to None. Defaults to `None`. |

**Methods:**

#### `select_range(text: str) → tuple[int, int]`

Select a proportion of text based on the anchor position.

| Parameter | Type | Description |
|---|---|---|
| `text` | `str` | The input text to select from. |

**Returns:**

- `tuple[int, int]` — tuple[int, int]: A tuple of (start_index, end_index).

## `class QRCodeConverter(PromptConverter)`

Converts a text string to a QR code image.

**Constructor Parameters:**

| Parameter | Type | Description |
|---|---|---|
| `scale` | `(int, Optional)` | Scaling factor that determines the width/height in pixels of each black/white square (known as a "module") in the QR code. Defaults to 3. Defaults to `3`. |
| `border` | `(int, Optional)` | Controls how many modules thick the border should be. Defaults to recommended value of 4. Defaults to `4`. |
| `dark_color` | `(tuple, Optional)` | Sets color of dark modules, using RGB values. Defaults to black: (0, 0, 0). Defaults to `(0, 0, 0)`. |
| `light_color` | `(tuple, Optional)` | Sets color of light modules, using RGB values. Defaults to white: (255, 255, 255). Defaults to `(255, 255, 255)`. |
| `data_dark_color` | `(tuple, Optional)` | Sets color of dark data modules (the modules that actually stores the data), using RGB values. Defaults to ``dark_color``. Defaults to `None`. |
| `data_light_color` | `(tuple, Optional)` | Sets color of light data modules, using RGB values. Defaults to light_color. Defaults to `None`. |
| `finder_dark_color` | `(tuple, Optional)` | Sets dark module color of finder patterns (squares located in three corners), using RGB values. Defaults to ``dark_color``. Defaults to `None`. |
| `finder_light_color` | `(tuple, Optional)` | Sets light module color of finder patterns, using RGB values. Defaults to light_color. Defaults to `None`. |
| `border_color` | `(tuple, Optional)` | Sets color of border, using RGB values. Defaults to ``light_color``. Defaults to `None`. |

**Methods:**

#### convert_async

```python
convert_async(prompt: str, input_type: PromptDataType = 'text') → ConverterResult
```

Convert the given prompt to a QR code image.

| Parameter | Type | Description |
|---|---|---|
| `prompt` | `str` | The prompt to be converted. |
| `input_type` | `PromptDataType` | The type of input data. Defaults to `'text'`. |

**Returns:**

- `ConverterResult` — The result containing filename of the converted QR code image.

**Raises:**

- `ValueError` — If the input type is not supported or if the prompt is empty.

## `class ROT13Converter(WordLevelConverter)`

Encodes prompts using the ROT13 cipher.

**Methods:**

#### `convert_word_async(word: str) → str`

Convert a single word into the target format supported by the converter.

| Parameter | Type | Description |
|---|---|---|
| `word` | `str` | The word to be converted. |

**Returns:**

- `str` — The converted word.

## `class RandomCapitalLettersConverter(PromptConverter)`

Takes a prompt and randomly capitalizes it by a percentage of the total characters.

**Constructor Parameters:**

| Parameter | Type | Description |
|---|---|---|
| `percentage` | `float` | The percentage of characters to capitalize in the prompt. Must be between 1 and 100. Defaults to 100.0. This includes decimal points in that range. Defaults to `100.0`. |

**Methods:**

#### convert_async

```python
convert_async(prompt: str, input_type: PromptDataType = 'text') → ConverterResult
```

Convert the given prompt by randomly capitalizing a percentage of its characters.

| Parameter | Type | Description |
|---|---|---|
| `prompt` | `str` | The input text prompt to be converted. |
| `input_type` | `PromptDataType` | The type of input data. Defaults to `'text'`. |

**Returns:**

- `ConverterResult` — The result containing the converted text.

**Raises:**

- `ValueError` — If the input type is not supported.

#### `generate_random_positions(total_length: int, set_number: int) → list[int]`

Generate a list of unique random positions within the range of `total_length`.

| Parameter | Type | Description |
|---|---|---|
| `total_length` | `int` | The total length of the string. |
| `set_number` | `int` | The number of unique random positions to generate. |

**Returns:**

- `list[int]` — A list of unique random positions.

**Raises:**

- `ValueError` — If `set_number` is greater than `total_length`.

#### `is_percentage(input_string: float) → bool`

Check if the input string is a valid percentage between 1 and 100.

| Parameter | Type | Description |
|---|---|---|
| `input_string` | `str` | The input string to check. |

**Returns:**

- `bool` — True if the input string is a valid percentage, False otherwise.

#### `string_to_upper_case_by_percentage(percentage: float, prompt: str) → str`

Convert a string by randomly capitalizing a percentage of its characters.

| Parameter | Type | Description |
|---|---|---|
| `percentage` | `float` | The percentage of characters to capitalize. |
| `prompt` | `str` | The input string to be converted. |

**Returns:**

- `str` — The converted string with randomly capitalized characters.

**Raises:**

- `ValueError` — If the percentage is not between 1 and 100.

## `class RandomTranslationConverter(LLMGenericTextConverter, WordLevelConverter)`

Translates each individual word in a prompt to a random language using an LLM.

An existing ``PromptChatTarget`` is used to perform the translation (like Azure OpenAI).

**Constructor Parameters:**

| Parameter | Type | Description |
|---|---|---|
| `converter_target` | `PromptChatTarget` | The target for the prompt conversion. Can be omitted if a default has been configured via PyRIT initialization. Defaults to `REQUIRED_VALUE`. |
| `system_prompt_template` | `Optional[SeedPrompt]` | The system prompt template to use for the conversion. If not provided, a default template will be used. Defaults to `None`. |
| `languages` | `Optional[List[str]]` | The list of available languages to use for translation. Defaults to `None`. |
| `word_selection_strategy` | `Optional[WordSelectionStrategy]` | Strategy for selecting which words to convert. If None, all words will be converted. Defaults to `None`. |

**Methods:**

#### convert_async

```python
convert_async(prompt: str, input_type: PromptDataType = 'text') → ConverterResult
```

Convert the given prompt into the target format supported by the converter.

| Parameter | Type | Description |
|---|---|---|
| `prompt` | `str` | The prompt to be converted. |
| `input_type` | `PromptDataType` | The type of input data. Defaults to `'text'`. |

**Returns:**

- `ConverterResult` — The result containing the converted output and its type.

#### `convert_word_async(word: str) → str`

Convert a single word into the target format supported by the converter.

| Parameter | Type | Description |
|---|---|---|
| `word` | `str` | The word to be converted. |

**Returns:**

- `str` — The converted word.

## `class RangeSelectionStrategy(TextSelectionStrategy)`

Selects text based on proportional start and end positions.

**Constructor Parameters:**

| Parameter | Type | Description |
|---|---|---|
| `start_proportion` | `float` | The starting position as a proportion (0.0 to 1.0). Defaults to 0.0. Defaults to `0.0`. |
| `end_proportion` | `float` | The ending position as a proportion (0.0 to 1.0). Defaults to 1.0. Defaults to `1.0`. |

**Methods:**

#### `select_range(text: str) → tuple[int, int]`

Select a range based on proportional positions.

| Parameter | Type | Description |
|---|---|---|
| `text` | `str` | The input text to select from. |

**Returns:**

- `tuple[int, int]` — tuple[int, int]: A tuple of (start_index, end_index).

## `class RegexSelectionStrategy(TextSelectionStrategy)`

Selects text based on the first regex match.

**Constructor Parameters:**

| Parameter | Type | Description |
|---|---|---|
| `pattern` | `Union[str, Pattern[str]]` | The regex pattern to match. |

**Methods:**

#### `select_range(text: str) → tuple[int, int]`

Select the range of the first regex match.

| Parameter | Type | Description |
|---|---|---|
| `text` | `str` | The input text to select from. |

**Returns:**

- `tuple[int, int]` — tuple[int, int]: A tuple of (start_index, end_index) of the first match,
or (0, 0) if no match found.

## `class RepeatTokenConverter(PromptConverter)`

Repeats a specified token a specified number of times in addition to a given prompt.

Based on:
https://dropbox.tech/machine-learning/bye-bye-bye-evolution-of-repeated-token-attacks-on-chatgpt-models

**Constructor Parameters:**

| Parameter | Type | Description |
|---|---|---|
| `token_to_repeat` | `str` | The string to be repeated. |
| `times_to_repeat` | `int` | The number of times the string will be repeated. |
| `token_insert_mode` | `str` | The mode of insertion for the repeated token. Can be "split", "prepend", "append", or "repeat". Defaults to `None`. |

**Methods:**

#### convert_async

```python
convert_async(prompt: str, input_type: PromptDataType = 'text') → ConverterResult
```

Convert the given prompt by repeating the specified token a specified number of times.

| Parameter | Type | Description |
|---|---|---|
| `prompt` | `str` | The prompt to be converted. |
| `input_type` | `PromptDataType` | The type of the input prompt. Defaults to `'text'`. |

**Returns:**

- `ConverterResult` — The result containing the modified prompt with repeated tokens.

**Raises:**

- `ValueError` — If the input type is not supported.

## `class ScientificTranslationConverter(LLMGenericTextConverter)`

Uses an LLM to transform simple or direct prompts into
scientifically-framed versions using technical terminology,
chemical notation, or academic phrasing.
This can be useful for red-teaming scenarios to test
whether safety filters can be bypassed through scientific translation.

**Constructor Parameters:**

| Parameter | Type | Description |
|---|---|---|
| `converter_target` | `PromptChatTarget` | The LLM target to perform the conversion. Defaults to `REQUIRED_VALUE`. |
| `mode` | `str` | The translation mode to use. Built-in options are:  - ``academic``: Use academic/homework style framing - ``technical``: Use technical jargon and terminology - ``smiles``: Uses chemical notation (e.g., SMILES or IUPAC notation such as   "2-(acetyloxy)benzoic acid" or "CC(=O)Oc1ccccc1C(=O)O" for aspirin) - ``research``: Frame as research/safety study or question - ``reaction``: Frame as a step-by-step chemistry mechanism problem - ``math``: Frame as the answer key to a mathematical problem or equation   for a homework/exam setting - ``combined``: Use combination of above techniques together (default)  You can also use a custom mode name if you provide a prompt_template. Defaults to `'combined'`. |
| `prompt_template` | `(SeedPrompt, Optional)` | Custom prompt template. Required if using a custom mode not in the built-in list. Defaults to `None`. |

## `class SearchReplaceConverter(PromptConverter)`

Converts a string by replacing chosen phrase with a new phrase of choice.

**Constructor Parameters:**

| Parameter | Type | Description |
|---|---|---|
| `pattern` | `str` | The regex pattern to replace. |
| `replace` | `str | list[str]` | The new phrase to replace with. Can be a single string or a list of strings. If a list is provided, a random element will be chosen for replacement. |
| `regex_flags` | `int` | Regex flags to use for the replacement. Defaults to 0 (no flags). Defaults to `0`. |

**Methods:**

#### convert_async

```python
convert_async(prompt: str, input_type: PromptDataType = 'text') → ConverterResult
```

Convert the given prompt by replacing the specified pattern with a random choice from the replacement list.

| Parameter | Type | Description |
|---|---|---|
| `prompt` | `str` | The prompt to be converted. |
| `input_type` | `PromptDataType` | The type of input data. Defaults to `'text'`. |

**Returns:**

- `ConverterResult` — The result containing the converted text.

**Raises:**

- `ValueError` — If the input type is not supported.

## `class SelectiveTextConverter(PromptConverter)`

A wrapper converter that applies another converter to selected portions of text.

This converter supports multiple selection strategies:
- Character-level: Selects a contiguous character range (e.g., IndexSelectionStrategy, RegexSelectionStrategy)
- Word-level: Selects specific words (e.g., WordIndexSelectionStrategy, WordPositionSelectionStrategy)
- Token-based: Auto-detects and converts text between ⟪⟫ tokens (TokenSelectionStrategy)

Most use cases will use word-level strategies for more intuitive selection.

**Constructor Parameters:**

| Parameter | Type | Description |
|---|---|---|
| `converter` | `PromptConverter` | The converter to apply to the selected text. |
| `selection_strategy` | `TextSelectionStrategy` | The strategy for selecting which text to convert. Can be character-level or word-level strategy. |
| `preserve_tokens` | `bool` | If True, wraps converted text with start/end tokens. This allows subsequent converters in a chain to target different regions. Defaults to False. Defaults to `False`. |
| `start_token` | `str` | The token to place before converted text when preserve_tokens=True. Defaults to "⟪". Defaults to `'⟪'`. |
| `end_token` | `str` | The token to place after converted text when preserve_tokens=True. Defaults to "⟫". Defaults to `'⟫'`. |
| `word_separator` | `str` | The separator to use when working with word-level strategies. Defaults to " ". Defaults to `' '`. |

**Methods:**

#### convert_async

```python
convert_async(prompt: str, input_type: PromptDataType = 'text') → ConverterResult
```

Convert selected portions of the prompt using the wrapped converter.

| Parameter | Type | Description |
|---|---|---|
| `prompt` | `str` | The prompt to be converted. |
| `input_type` | `PromptDataType` | The type of input data. Must be "text". Defaults to `'text'`. |

**Returns:**

- `ConverterResult` — The result containing the converted output and its type.

**Raises:**

- `ValueError` — If the input type is not "text".

## `class SneakyBitsSmugglerConverter(SmugglerConverter)`

Encodes and decodes text using a bit-level approach.

**Constructor Parameters:**

| Parameter | Type | Description |
|---|---|---|
| `action` | `Literal['encode', 'decode']` | The action to perform. Defaults to `'encode'`. |
| `zero_char` | `Optional[str]` | Character to represent binary 0 in ``sneaky_bits`` mode (default: U+2062). Defaults to `None`. |
| `one_char` | `Optional[str]` | Character to represent binary 1 in ``sneaky_bits`` mode (default: U+2064). Defaults to `None`. |

**Methods:**

#### `decode_message(message: str) → str`

Decode the message encoded using Sneaky Bits mode.

The method filters out only the valid invisible characters (``self.zero_char`` and ``self.one_char``),
groups them into 8-bit chunks, reconstructs each byte, and finally decodes the byte sequence using UTF-8.

| Parameter | Type | Description |
|---|---|---|
| `message` | `str` | The message encoded with Sneaky Bits. |

**Returns:**

- `str` — The decoded original message.

#### `encode_message(message: str) → tuple[str, str]`

Encode the message using Sneaky Bits mode.

The message is first converted to its UTF-8 byte sequence. Then each byte is represented as 8 bits,
with each bit replaced by an invisible character (``self.zero_char`` for 0 and ``self.one_char`` for 1).

| Parameter | Type | Description |
|---|---|---|
| `message` | `str` | The message to encode. |

**Returns:**

- `str` — Tuple[str, str]: A tuple where the first element is a bit summary (empty in this implementation)
- `str` — and the second element is the encoded message containing the invisible bits.

## `class StringJoinConverter(WordLevelConverter)`

Converts text by joining its characters with the specified join value.

**Constructor Parameters:**

| Parameter | Type | Description |
|---|---|---|
| `join_value` | `str` | The string used to join characters of each word. Defaults to `'-'`. |
| `word_selection_strategy` | `Optional[WordSelectionStrategy]` | Strategy for selecting which words to convert. If None, all words will be converted. Defaults to `None`. |

**Methods:**

#### `convert_word_async(word: str) → str`

Convert a single word into the target format supported by the converter.

| Parameter | Type | Description |
|---|---|---|
| `word` | `str` | The word to be converted. |

**Returns:**

- `str` — The converted word.

## `class SuffixAppendConverter(PromptConverter)`

Appends a specified suffix to the prompt.
E.g. with a suffix `!!!`, it converts a prompt of `test` to `test !!!`.

See https://github.com/Azure/PyRIT/tree/main/pyrit/auxiliary_attacks/gcg for adversarial suffix generation.

**Constructor Parameters:**

| Parameter | Type | Description |
|---|---|---|
| `suffix` | `str` | The suffix to append to the prompt. |

**Methods:**

#### convert_async

```python
convert_async(prompt: str, input_type: PromptDataType = 'text') → ConverterResult
```

Convert the given prompt by appending the specified suffix.

| Parameter | Type | Description |
|---|---|---|
| `prompt` | `str` | The prompt to be converted. |
| `input_type` | `PromptDataType` | Type of input data. Defaults to "text". Defaults to `'text'`. |

**Returns:**

- `ConverterResult` — The prompt with the suffix appended.

**Raises:**

- `ValueError` — If the input type is not supported.

## `class SuperscriptConverter(WordLevelConverter)`

Converts text to superscript.

This converter leaves characters that do not have a superscript equivalent unchanged.

**Methods:**

#### `convert_word_async(word: str) → str`

Convert a single word into the target format supported by the converter.

| Parameter | Type | Description |
|---|---|---|
| `word` | `str` | The word to be converted. |

**Returns:**

- `str` — The converted word.

## `class TemplateSegmentConverter(PromptConverter)`

Uses a template to randomly split a prompt into segments defined by the template.

This converter is a generalized version of this:
https://adversa.ai/blog/universal-llm-jailbreak-chatgpt-gpt-4-bard-bing-anthropic-and-beyond/

**Constructor Parameters:**

| Parameter | Type | Description |
|---|---|---|
| `prompt_template` | `(SeedPrompt, Optional)` | The prompt template for the conversion. Must have two or more parameters. If not provided, uses the default ``tom_and_jerry.yaml`` template. Defaults to `None`. |

**Methods:**

#### convert_async

```python
convert_async(prompt: str, input_type: PromptDataType = 'text') → ConverterResult
```

Convert the given prompt by splitting it into random segments and using them to fill the template parameters.
The prompt is split into N segments (where N is the number of template parameters) at random word boundaries.
Each segment is then used to fill the corresponding template parameter.

| Parameter | Type | Description |
|---|---|---|
| `prompt` | `str` | The prompt to be converted. |
| `input_type` | `PromptDataType` | The type of input data. Defaults to `'text'`. |

**Returns:**

- `ConverterResult` — The result containing the template filled with prompt segments.

**Raises:**

- `ValueError` — If the input type is not supported.

## `class TenseConverter(LLMGenericTextConverter)`

Converts a conversation to a different tense using an LLM.

An existing ``PromptChatTarget`` is used to perform the conversion (like Azure OpenAI).

**Constructor Parameters:**

| Parameter | Type | Description |
|---|---|---|
| `converter_target` | `PromptChatTarget` | The target chat support for the conversion which will translate. Can be omitted if a default has been configured via PyRIT initialization. Defaults to `REQUIRED_VALUE`. |
| `tense` | `str` | The tense the converter should convert the prompt to. E.g. past, present, future. |
| `prompt_template` | `(SeedPrompt, Optional)` | The prompt template for the conversion. Defaults to `None`. |

## `class TextJailbreakConverter(PromptConverter)`

Uses a jailbreak template to create a prompt.

**Constructor Parameters:**

| Parameter | Type | Description |
|---|---|---|
| `jailbreak_template` | `TextJailBreak` | The jailbreak template to use for conversion. |

**Methods:**

#### convert_async

```python
convert_async(prompt: str, input_type: PromptDataType = 'text') → ConverterResult
```

Convert the given prompt using the jailbreak template.

| Parameter | Type | Description |
|---|---|---|
| `prompt` | `str` | The prompt to be converted. |
| `input_type` | `PromptDataType` | The type of input data. Defaults to `'text'`. |

**Returns:**

- `ConverterResult` — The result containing the converted output and its type.

**Raises:**

- `ValueError` — If the input type is not supported.

## `class TextSelectionStrategy(abc.ABC)`

Base class for text selection strategies used by SelectiveTextConverter and WordLevelConverter.
Defines how to select a region of text or words for conversion.

**Methods:**

#### `select_range(text: str) → tuple[int, int]`

Select a range of characters in the text to be converted.

| Parameter | Type | Description |
|---|---|---|
| `text` | `str` | The input text to select from. |

**Returns:**

- `tuple[int, int]` — tuple[int, int]: A tuple of (start_index, end_index) representing the character range.
The range is inclusive of start_index and exclusive of end_index.

## `class TokenSelectionStrategy(TextSelectionStrategy)`

A special selection strategy that signals SelectiveTextConverter to auto-detect
and convert text between start/end tokens (e.g., ⟪ and ⟫).

This strategy is used when chaining converters with preserve_tokens=True.
Instead of programmatically selecting text, it relies on tokens already present
in the text from a previous converter.

**Methods:**

#### `select_range(text: str) → tuple[int, int]`

Do not use this method for TokenSelectionStrategy.
SelectiveTextConverter handles token detection separately.

| Parameter | Type | Description |
|---|---|---|
| `text` | `str` | The input text (ignored). |

**Returns:**

- `tuple[int, int]` — tuple[int, int]: Always returns (0, 0) as this strategy uses token detection instead.

## `class ToneConverter(LLMGenericTextConverter)`

Converts a conversation to a different tone using an LLM.

An existing ``PromptChatTarget`` is used to perform the conversion (like Azure OpenAI).

**Constructor Parameters:**

| Parameter | Type | Description |
|---|---|---|
| `converter_target` | `PromptChatTarget` | The target chat support for the conversion which will translate. Can be omitted if a default has been configured via PyRIT initialization. Defaults to `REQUIRED_VALUE`. |
| `tone` | `str` | The tone for the conversation. E.g. upset, sarcastic, indifferent, etc. |
| `prompt_template` | `(SeedPrompt, Optional)` | The prompt template for the conversion. Defaults to `None`. |

## `class ToxicSentenceGeneratorConverter(LLMGenericTextConverter)`

Generates toxic sentence starters using an LLM.

An existing ``PromptChatTarget`` is used to perform the conversion (like Azure OpenAI).

Based on Project Moonshot's attack module that generates toxic sentences to test LLM
safety guardrails:
https://github.com/aiverify-foundation/moonshot-data/blob/main/attack-modules/toxic_sentence_generator.py

**Constructor Parameters:**

| Parameter | Type | Description |
|---|---|---|
| `converter_target` | `PromptChatTarget` | The endpoint that converts the prompt. Can be omitted if a default has been configured via PyRIT initialization. Defaults to `REQUIRED_VALUE`. |
| `prompt_template` | `SeedPrompt` | The seed prompt template to use. If not provided,                           defaults to the ``toxic_sentence_generator.yaml``. Defaults to `None`. |

**Methods:**

#### convert_async

```python
convert_async(prompt: str, input_type: PromptDataType = 'text') → ConverterResult
```

Convert the given prompt into a toxic sentence starter.

| Parameter | Type | Description |
|---|---|---|
| `prompt` | `str` | The prompt to be converted. |
| `input_type` | `PromptDataType` | The type of input data. Defaults to `'text'`. |

**Returns:**

- `ConverterResult` — The conversion result containing the toxic sentence starter.

#### `input_supported(input_type: PromptDataType) → bool`

Check if the input type is supported.

| Parameter | Type | Description |
|---|---|---|
| `input_type` | `PromptDataType` | The type of input data. |

**Returns:**

- `bool` — True if the input type is supported, False otherwise.

#### `output_supported(output_type: PromptDataType) → bool`

Check if the output type is supported.

| Parameter | Type | Description |
|---|---|---|
| `output_type` | `PromptDataType` | The type of output data. |

**Returns:**

- `bool` — True if the output type is supported, False otherwise.

## `class TranslationConverter(PromptConverter)`

Translates prompts into different languages using an LLM.

**Constructor Parameters:**

| Parameter | Type | Description |
|---|---|---|
| `converter_target` | `PromptChatTarget` | The target chat support for the conversion which will translate. Can be omitted if a default has been configured via PyRIT initialization. Defaults to `REQUIRED_VALUE`. |
| `language` | `str` | The language for the conversion. E.g. Spanish, French, leetspeak, etc. |
| `prompt_template` | `(SeedPrompt, Optional)` | The prompt template for the conversion. Defaults to `None`. |
| `max_retries` | `int` | Maximum number of retries for the conversion. Defaults to `3`. |
| `max_wait_time_in_seconds` | `int` | Maximum wait time in seconds between retries. Defaults to `60`. |

**Methods:**

#### convert_async

```python
convert_async(prompt: str, input_type: PromptDataType = 'text') → ConverterResult
```

Convert the given prompt by translating it using the converter target.

| Parameter | Type | Description |
|---|---|---|
| `prompt` | `str` | The prompt to be converted. |
| `input_type` | `PromptDataType` | The type of input data. Defaults to `'text'`. |

**Returns:**

- `ConverterResult` — The result containing the generated version of the prompt.

**Raises:**

- `ValueError` — If the input type is not supported.

## `class TransparencyAttackConverter(PromptConverter)`

Currently, only JPEG images are supported as input. Output images will always be saved as PNG with transparency.

**Constructor Parameters:**

| Parameter | Type | Description |
|---|---|---|
| `benign_image_path` | `Path` | Path to the benign image file. Must be a JPEG file (.jpg or .jpeg). |
| `size` | `tuple` | Size that the images will be resized to (width, height). It is recommended to use a size that matches aspect ratio of both attack and benign images. Since the original study resizes images to 150x150 pixels, this is the default size used. Bigger values may significantly increase computation time. Defaults to `(150, 150)`. |
| `steps` | `int` | Number of optimization steps to perform. Recommended range: 100-2000 steps. Default is 1500. Generally, the higher the steps, the better end result you can achieve, but at the cost of increased computation time. Defaults to `1500`. |
| `learning_rate` | `float` | Controls the magnitude of adjustments in each step (used by the Adam optimizer). Recommended range: 0.0001-0.01. Default is 0.001. Values close to 1 may lead to instability and lower quality blending, while values too low may require more steps to achieve a good blend. Defaults to `0.001`. |
| `convergence_threshold` | `float` | Minimum change in loss required to consider improvement. If the change in loss between steps is below this value, it's counted as no improvement. Default is 1e-6. Recommended range: 1e-6 to 1e-4. Defaults to `1e-06`. |
| `convergence_patience` | `int` | Number of consecutive steps with no improvement before stopping. Default is 10. Defaults to `10`. |

**Methods:**

#### convert_async

```python
convert_async(prompt: str, input_type: PromptDataType = 'image_path') → ConverterResult
```

Convert the given prompt by blending an attack image (potentially harmful) with a benign image.
Uses the Novel Image Blending Algorithm from: https://arxiv.org/abs/2401.15817.

| Parameter | Type | Description |
|---|---|---|
| `prompt` | `str` | The image file path to the attack image. |
| `input_type` | `PromptDataType` | The type of input data. Must be "image_path". Defaults to `'image_path'`. |

**Returns:**

- `ConverterResult` — The result containing path to the manipulated image with transparency.

**Raises:**

- `ValueError` — If the input type is not supported or if the prompt is invalid.

## `class UnicodeConfusableConverter(PromptConverter)`

Applies substitutions to words in the prompt to test adversarial textual robustness
by replacing characters with visually similar ones.

**Constructor Parameters:**

| Parameter | Type | Description |
|---|---|---|
| `source_package` | `Literal['confusable_homoglyphs', 'confusables']` |  The package to use for homoglyph generation.  Can be either:     - "confusable_homoglyphs" (https://pypi.org/project/confusable-homoglyphs/):         Used by default as it is more regularly maintained and up to date with the latest         Unicode-provided confusables found here:         https://www.unicode.org/Public/security/latest/confusables.txt     - "confusables" (https://pypi.org/project/confusables/):         Provides additional methods of matching characters (not just Unicode list),         so each character has more possible substitutions. Defaults to `'confusable_homoglyphs'`. |
| `deterministic` | `bool` | This argument is for unittesting only. Defaults to `False`. |

**Methods:**

#### convert_async

```python
convert_async(prompt: str, input_type: PromptDataType = 'text') → ConverterResult
```

Convert the given prompt by applying confusable substitutions. This leads to a prompt that looks similar,
but is actually different (e.g., replacing a Latin 'a' with a Cyrillic 'а').

| Parameter | Type | Description |
|---|---|---|
| `prompt` | `str` | The prompt to be converted. |
| `input_type` | `PromptDataType` | The type of input data. Defaults to `'text'`. |

**Returns:**

- `ConverterResult` — The result containing the prompt with confusable substitutions applied.

**Raises:**

- `ValueError` — If the input type is not supported.

## `class UnicodeReplacementConverter(WordLevelConverter)`

Converts a prompt to its unicode representation.

**Constructor Parameters:**

| Parameter | Type | Description |
|---|---|---|
| `encode_spaces` | `bool` | If True, spaces in the prompt will be replaced with unicode representation. Defaults to `False`. |
| `word_selection_strategy` | `Optional[WordSelectionStrategy]` | Strategy for selecting which words to convert. If None, all words will be converted. Defaults to `None`. |

**Methods:**

#### `convert_word_async(word: str) → str`

Convert a single word into the target format supported by the converter.

| Parameter | Type | Description |
|---|---|---|
| `word` | `str` | The word to be converted. |

**Returns:**

- `str` — The converted word.

#### `join_words(words: list[str]) → str`

Join a list of words into a single string, optionally encoding spaces as unicode.

| Parameter | Type | Description |
|---|---|---|
| `words` | `list[str]` | The list of words to join. |

**Returns:**

- `str` — The joined string.

## `class UnicodeSubstitutionConverter(PromptConverter)`

Encodes the prompt using any unicode starting point.

**Constructor Parameters:**

| Parameter | Type | Description |
|---|---|---|
| `start_value` | `int` | The unicode starting point to use for encoding. Defaults to `917504`. |

**Methods:**

#### convert_async

```python
convert_async(prompt: str, input_type: PromptDataType = 'text') → ConverterResult
```

Convert the given prompt by encoding it using any unicode starting point.
Default is to use invisible flag emoji characters.

| Parameter | Type | Description |
|---|---|---|
| `prompt` | `str` | The prompt to be converted. |
| `input_type` | `PromptDataType` | The type of input data. Defaults to `'text'`. |

**Returns:**

- `ConverterResult` — The result containing the converted output and its type.

**Raises:**

- `ValueError` — If the input type is not supported.

## `class UrlConverter(PromptConverter)`

Converts a prompt to a URL-encoded string.

**Methods:**

#### convert_async

```python
convert_async(prompt: str, input_type: PromptDataType = 'text') → ConverterResult
```

Convert the given prompt into a URL-encoded string.

| Parameter | Type | Description |
|---|---|---|
| `prompt` | `str` | The prompt to be converted. |
| `input_type` | `PromptDataType` | The type of input data. Defaults to `'text'`. |

**Returns:**

- `ConverterResult` — The result containing the URL-encoded prompt.

**Raises:**

- `ValueError` — If the input type is not supported.

## `class VariationConverter(PromptConverter)`

Generates variations of the input prompts using the converter target.

**Constructor Parameters:**

| Parameter | Type | Description |
|---|---|---|
| `converter_target` | `PromptChatTarget` | The target to which the prompt will be sent for conversion. Can be omitted if a default has been configured via PyRIT initialization. Defaults to `REQUIRED_VALUE`. |
| `prompt_template` | `SeedPrompt` | The template used for generating the system prompt. If not provided, a default template will be used. Defaults to `None`. |

**Methods:**

#### convert_async

```python
convert_async(prompt: str, input_type: PromptDataType = 'text') → ConverterResult
```

Convert the given prompt by generating variations of it using the converter target.

| Parameter | Type | Description |
|---|---|---|
| `prompt` | `str` | The prompt to be converted. |
| `input_type` | `PromptDataType` | The type of input data. Defaults to `'text'`. |

**Returns:**

- `ConverterResult` — The result containing the generated variations.

**Raises:**

- `ValueError` — If the input type is not supported.

#### `send_variation_prompt_async(request: Message) → str`

Send the message to the converter target and retrieve the response.

| Parameter | Type | Description |
|---|---|---|
| `request` | `Message` | The message to be sent to the converter target. |

**Returns:**

- `str` — The response message from the converter target.

**Raises:**

- `InvalidJsonException` — If the response is not valid JSON or does not contain the expected keys.

## `class VariationSelectorSmugglerConverter(SmugglerConverter)`

Extension: In addition to embedding into a base character, we also support
appending invisible variation selectors directly to visible text—enabling mixed
visible and hidden content within a single string.

**Constructor Parameters:**

| Parameter | Type | Description |
|---|---|---|
| `action` | `Literal['encode', 'decode']` | The action to perform. Defaults to `'encode'`. |
| `base_char_utf8` | `Optional[str]` | Base character for ``variation_selector_smuggler`` mode (default: 😊). Defaults to `None`. |
| `embed_in_base` | `bool` | If True, the hidden payload is embedded directly into the base character.                     If False, a visible separator (space) is inserted between the base and payload.                     Default is True. Defaults to `True`. |

**Methods:**

#### `decode_message(message: str) → str`

Decode a message encoded using Unicode variation selectors.
The decoder scans the string for variation selectors, ignoring any visible separator.

| Parameter | Type | Description |
|---|---|---|
| `message` | `str` | The encoded message. |

**Returns:**

- `str` — The decoded message.

#### `decode_visible_hidden(combined: str) → tuple[str, str]`

Extract the visible text and decodes the hidden text from a combined string.

It searches for the first occurrence of the base character (``self.utf8_base_char``) and treats everything
from that point on as the hidden payload.

| Parameter | Type | Description |
|---|---|---|
| `combined` | `str` | The combined text containing visible and hidden parts. |

**Returns:**

- `tuple[str, str]` — Tuple[str, str]: A tuple with the visible text and the decoded hidden text.

#### `encode_message(message: str) → tuple[str, str]`

Encode the message using Unicode variation selectors.

The message is converted to UTF-8 bytes, and each byte is mapped to a variation selector:
    - 0x00-0x0F => U+FE00 to U+FE0F.
    - 0x10-0xFF => U+E0100 to U+E01EF.

If ``embed_in_base`` is True, the payload is embedded directly into the base character;
otherwise, a visible separator (a space) is inserted between the base and payload.

| Parameter | Type | Description |
|---|---|---|
| `message` | `str` | The message to encode. |

**Returns:**

- `tuple[str, str]` — Tuple[str, str]: A tuple containing a summary of the code points and the encoded string.

#### `encode_visible_hidden(visible: str, hidden: str) → tuple[str, str]`

Combine visible text with hidden text by encoding the hidden text using ``variation_selector_smuggler`` mode.

The hidden payload is generated as a composite using the current embedding setting and then appended
to the visible text.

| Parameter | Type | Description |
|---|---|---|
| `visible` | `str` | The visible text. |
| `hidden` | `str` | The secret/hidden text to encode. |

**Returns:**

- `tuple[str, str]` — Tuple[str, str]: A tuple containing a summary and the combined text.

## `class WordDocConverter(PromptConverter)`

Convert a text prompt into a Word (.docx) document.

This converter supports two main modes:

1. **New document generation**
   If no existing document is provided, the converter creates a simple `.docx`
   containing the rendered prompt content in a single paragraph.

2. **Placeholder-based injection into an existing document**
   If an ``existing_docx`` is provided, the converter searches for a literal
   placeholder string (for example ``{{INJECTION_PLACEHOLDER}}``) in the
   document's paragraphs. When the placeholder is found fully inside a single
   run, it is replaced with the rendered prompt content while preserving the
   rest of the paragraph and its formatting.

   .. important::
      Placeholders must be fully contained within a single run. If a
      placeholder spans multiple runs (for example due to mixed formatting),
      this converter will not replace it. This limitation is intentional to
      avoid collapsing mixed formatting or rewriting complex run structures.

**Constructor Parameters:**

| Parameter | Type | Description |
|---|---|---|
| `prompt_template` | `Optional[SeedPrompt]` | Optional ``SeedPrompt`` template used to render the final content before injection. If provided, ``prompt`` passed to ``convert_async`` must be a string whose contents can be interpreted as the template parameters (for example, a JSON-encoded or other parseable mapping of keys to values). Defaults to `None`. |
| `existing_docx` | `Optional[Path]` | Optional path to an existing `.docx` file. When provided, the converter will search for ``placeholder`` inside the document paragraphs and replace it with the rendered content. If not provided, a new document is generated instead. Defaults to `None`. |
| `placeholder` | `str` | Literal placeholder text to search for in the existing document. This value must be fully contained within a single run for the replacement to succeed. Defaults to `'{{INJECTION_PLACEHOLDER}}'`. |

**Methods:**

#### convert_async

```python
convert_async(prompt: str, input_type: PromptDataType = 'text') → ConverterResult
```

Convert the given prompt into a Word document (.docx).

If ``prompt_template`` is provided, the prompt is first used to render the
template via ``SeedPrompt.render_template_value``. Otherwise, the raw
``prompt`` string is used as the content.

- When ``existing_docx`` is set, this content is injected into the
  document by replacing the configured placeholder string.
- When no ``existing_docx`` is provided, a new document with a single
  paragraph containing the content is created.

| Parameter | Type | Description |
|---|---|---|
| `prompt` | `str` | The prompt or dynamic data used to generate the content. |
| `input_type` | `PromptDataType` | The type of input data. Must be ``"text"``. Defaults to `'text'`. |

**Returns:**

- `ConverterResult` — Contains the path to the generated `.docx` file in
- `ConverterResult` — ``output_text`` and ``output_type="binary_path"``.

**Raises:**

- `ValueError` — If the input type is not supported.

## `class WordIndexSelectionStrategy(WordSelectionStrategy)`

Selects words based on their indices in the word list.

**Constructor Parameters:**

| Parameter | Type | Description |
|---|---|---|
| `indices` | `List[int]` | The list of word indices to select. |

**Methods:**

#### `select_words(words: list[str]) → list[int]`

Select words at the specified indices.

| Parameter | Type | Description |
|---|---|---|
| `words` | `List[str]` | The list of words to select from. |

**Returns:**

- `list[int]` — List[int]: The list of valid indices.

**Raises:**

- `ValueError` — If any indices are out of range.

## `class WordKeywordSelectionStrategy(WordSelectionStrategy)`

Selects words that match specific keywords.

**Constructor Parameters:**

| Parameter | Type | Description |
|---|---|---|
| `keywords` | `List[str]` | The list of keywords to match. |
| `case_sensitive` | `bool` | Whether matching is case-sensitive. Defaults to True. Defaults to `True`. |

**Methods:**

#### `select_words(words: list[str]) → list[int]`

Select words that match the keywords.

| Parameter | Type | Description |
|---|---|---|
| `words` | `List[str]` | The list of words to select from. |

**Returns:**

- `list[int]` — List[int]: The list of indices where keywords were found.

## `class WordPositionSelectionStrategy(WordSelectionStrategy)`

Selects words based on proportional start and end positions.

**Constructor Parameters:**

| Parameter | Type | Description |
|---|---|---|
| `start_proportion` | `float` | The starting position as a proportion (0.0 to 1.0). |
| `end_proportion` | `float` | The ending position as a proportion (0.0 to 1.0). |

**Methods:**

#### `select_words(words: list[str]) → list[int]`

Select words based on the relative position.

| Parameter | Type | Description |
|---|---|---|
| `words` | `List[str]` | The list of words to select from. |

**Returns:**

- `list[int]` — List[int]: The list of indices in the specified position range.

## `class WordProportionSelectionStrategy(WordSelectionStrategy)`

Selects a random proportion of words.

**Constructor Parameters:**

| Parameter | Type | Description |
|---|---|---|
| `proportion` | `float` | The proportion of words to select (0.0 to 1.0). |
| `seed` | `Optional[int]` | Random seed for reproducible selections. Defaults to None. Defaults to `None`. |

**Methods:**

#### `select_words(words: list[str]) → list[int]`

Select a random proportion of words.

| Parameter | Type | Description |
|---|---|---|
| `words` | `List[str]` | The list of words to select from. |

**Returns:**

- `list[int]` — List[int]: The list of randomly selected indices.

## `class WordRegexSelectionStrategy(WordSelectionStrategy)`

Selects words that match a regex pattern.

**Constructor Parameters:**

| Parameter | Type | Description |
|---|---|---|
| `pattern` | `Union[str, Pattern[str]]` | The regex pattern to match against words. |

**Methods:**

#### `select_words(words: list[str]) → list[int]`

Select words that match the regex pattern.

| Parameter | Type | Description |
|---|---|---|
| `words` | `List[str]` | The list of words to select from. |

**Returns:**

- `list[int]` — List[int]: The list of indices where words matched the pattern.

## `class WordSelectionStrategy(TextSelectionStrategy)`

Base class for word-level selection strategies.

Word selection strategies work by splitting text into words and selecting specific word indices.
They provide a select_words() method and implement select_range() by converting word selections
to character ranges.

**Methods:**

#### `select_range(text: str, word_separator: str = ' ') → tuple[int, int]`

Select a character range by first selecting words, then converting to character positions.

This implementation splits the text by word_separator, gets selected word indices,
then calculates the character range that spans those words.

| Parameter | Type | Description |
|---|---|---|
| `text` | `str` | The input text to select from. |
| `word_separator` | `str` | The separator used to split words. Defaults to " ". Defaults to `' '`. |

**Returns:**

- `tuple[int, int]` — tuple[int, int]: A tuple of (start_index, end_index) representing the character range
that encompasses all selected words.

#### `select_words(words: list[str]) → list[int]`

Select word indices to be converted.

| Parameter | Type | Description |
|---|---|---|
| `words` | `List[str]` | The list of words to select from. |

**Returns:**

- `list[int]` — List[int]: A list of indices representing which words should be converted.

## `class ZalgoConverter(WordLevelConverter)`

Converts text into cursed Zalgo text using combining Unicode marks.

**Constructor Parameters:**

| Parameter | Type | Description |
|---|---|---|
| `intensity` | `int` | Number of combining marks per character (higher = more cursed). Default is 10. Defaults to `10`. |
| `seed` | `Optional[int]` | Optional seed for reproducible output. Defaults to `None`. |
| `word_selection_strategy` | `Optional[WordSelectionStrategy]` | Strategy for selecting which words to convert. If None, all words will be converted. Defaults to `None`. |

**Methods:**

#### `convert_word_async(word: str) → str`

Convert a single word into the target format supported by the converter.

| Parameter | Type | Description |
|---|---|---|
| `word` | `str` | The word to be converted. |

**Returns:**

- `str` — The converted word.

#### `validate_input(prompt: str) → None`

Validate the input prompt before conversion.

## `class ZeroWidthConverter(PromptConverter)`

Injects zero-width spaces between characters in the provided text to bypass content safety mechanisms.

**Methods:**

#### convert_async

```python
convert_async(prompt: str, input_type: PromptDataType = 'text') → ConverterResult
```

Convert the given prompt by injecting zero-width spaces between each character.

| Parameter | Type | Description |
|---|---|---|
| `prompt` | `str` | The prompt to be converted. |
| `input_type` | `PromptDataType` | The type of input data. Defaults to `'text'`. |

**Returns:**

- `ConverterResult` — The result containing the modified prompt.

**Raises:**

- `ValueError` — If the input type is not supported.
