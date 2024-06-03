# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from pyrit.prompt_converter import (
    AddTextImageConverter,
    AsciiArtConverter,
    AzureSpeechTextToAudioConverter,
    Base64Converter,
    LeetspeakConverter,
    RandomCapitalLettersConverter,
    ROT13Converter,
    SearchReplaceConverter,
    StringJoinConverter,
    TranslationConverter,
    UnicodeSubstitutionConverter,
    UnicodeConfusableConverter,
    VariationConverter,
    SuffixAppendConverter,
    RepeatTokenConverter,
)
import pytest
import os

from tests.mocks import MockPromptTarget
from unittest.mock import patch, MagicMock
from PIL import Image, ImageFont
from io import BytesIO
import azure.cognitiveservices.speech as speechsdk


@pytest.mark.asyncio
async def test_base64_prompt_converter() -> None:
    converter = Base64Converter()
    output = await converter.convert_async(prompt="test", input_type="text")
    assert output.output_text == "dGVzdA=="
    assert output.output_type == "text"


@pytest.mark.asyncio
async def test_rot13_converter_init() -> None:
    converter = ROT13Converter()
    output = await converter.convert_async(prompt="test", input_type="text")
    assert output.output_text == "grfg"
    assert output.output_type == "text"


@pytest.mark.asyncio
async def test_unicode_sub_default_prompt_converter() -> None:
    converter = UnicodeSubstitutionConverter()
    output = await converter.convert_async(prompt="test", input_type="text")
    assert output.output_text == "\U000e0074\U000e0065\U000e0073\U000e0074"
    assert output.output_type == "text"


@pytest.mark.asyncio
async def test_unicode_sub_ascii_prompt_converter() -> None:
    converter = UnicodeSubstitutionConverter(start_value=0x00000)
    output = await converter.convert_async(prompt="test", input_type="text")
    assert output.output_text == "\U00000074\U00000065\U00000073\U00000074"
    assert output.output_type == "text"


@pytest.mark.asyncio
async def test_str_join_converter_default() -> None:
    converter = StringJoinConverter()
    output = await converter.convert_async(prompt="test", input_type="text")
    assert output.output_text == "t-e-s-t"
    assert output.output_type == "text"


@pytest.mark.asyncio
async def test_str_join_converter_init() -> None:
    converter = StringJoinConverter(join_value="***")
    output = await converter.convert_async(prompt="test", input_type="text")
    assert output.output_text == "t***e***s***t"
    assert output.output_type == "text"


@pytest.mark.asyncio
async def test_str_join_converter_none_raises() -> None:
    converter = StringJoinConverter()
    with pytest.raises(TypeError):
        assert await converter.convert_async(prompt=None, input_type="text")


@pytest.mark.asyncio
async def test_str_join_converter_invalid_type_raises() -> None:
    converter = StringJoinConverter()
    with pytest.raises(ValueError):
        assert await converter.convert_async(prompt="test", input_type="invalid")  # type: ignore # noqa


@pytest.mark.asyncio
async def test_str_join_converter_unsupported_type_raises() -> None:
    converter = StringJoinConverter()
    with pytest.raises(ValueError):
        assert await converter.convert_async(prompt="test", input_type="image_path")


@pytest.mark.asyncio
async def test_ascii_art() -> None:
    converter = AsciiArtConverter(font="block")
    output = await converter.convert_async(prompt="test", input_type="text")

    assert output.output_text == (
        "\n .----------------.  .----------------.  .----------------.  .----------------. \n| .--------------. || .--------------. || .--------------. || .--------------. |\n| |  _________   | || |  _________   | || |    _______   | || |  _________   | |\n| | |  _   _  |  | || | |_   ___  |  | || |   /  ___  |  | || | |  _   _  |  | |\n| | |_/ | | \\_|  | || |   | |_  \\_|  | || |  |  (__ \\_|  | || | |_/ | | \\_|  | |\n| |     | |      | || |   |  _|  _   | || |   '.___`-.   | || |     | |      | |\n| |    _| |_     | || |  _| |___/ |  | || |  |`\\____) |  | || |    _| |_     | |\n| |   |_____|    | || | |_________|  | || |  |_______.'  | || |   |_____|    | |\n| |              | || |              | || |              | || |              | |\n| '--------------' || '--------------' || '--------------' || '--------------' |\n '----------------'  '----------------'  '----------------'  '----------------' \n"  # noqa: E501
    )
    assert output.output_type == "text"


def test_prompt_variation_init_templates_not_null():
    prompt_target = MockPromptTarget()
    prompt_variation = VariationConverter(converter_target=prompt_target)
    assert prompt_variation.system_prompt


def test_prompt_translation_init_templates_not_null():
    prompt_target = MockPromptTarget()
    translation_converter = TranslationConverter(converter_target=prompt_target, language="en")
    assert translation_converter.system_prompt


@pytest.mark.parametrize("languages", [None, ""])
def test_translator_converter_languages_validation_throws(languages):
    prompt_target = MockPromptTarget()
    with pytest.raises(ValueError):
        TranslationConverter(converter_target=prompt_target, language=languages)


@pytest.mark.asyncio
async def test_unicode_confusable_converter() -> None:
    converter = UnicodeConfusableConverter(deterministic=True)
    output = await converter.convert_async(prompt="lorem ipsum dolor sit amet", input_type="text")
    assert output.output_text == "ïỎ𐒴ḕ𝗠 ïṗṡ𝘶𝗠 𝑫ỎïỎ𐒴 ṡï𝚝 ḁ𝗠ḕ𝚝"
    assert output.output_type == "text"


@pytest.mark.asyncio
async def test_character_replacement_converter() -> None:
    converter = SearchReplaceConverter(old_value=" ", new_value="_")
    output = await converter.convert_async(prompt="Hello World !", input_type="text")
    assert output.output_text == "Hello_World_!"
    assert output.output_type == "text"


@pytest.mark.asyncio
async def test_leetcode_converter() -> None:
    converter = LeetspeakConverter()
    output = await converter.convert_async(prompt="mood", input_type="text")
    assert output.output_text == "m00d"
    assert output.output_type == "text"


@pytest.mark.asyncio
async def test_capital_letter_converter() -> None:
    converter = RandomCapitalLettersConverter()
    output = await converter.convert_async(
        prompt="Write an article about cat documents that proved fraudulent, county Judge Orders cat documents"
        " need to be redone!",
        input_type="text",
    )

    assert (
        output.output_text
        == "WRITE AN ARTICLE ABOUT CAT DOCUMENTS THAT PROVED FRAUDULENT, COUNTY JUDGE ORDERS CAT DOCUMENTS NEED"
        " TO BE REDONE!"
    )


@pytest.mark.asyncio
async def test_suffix_append_converter() -> None:
    converter = SuffixAppendConverter(suffix="!!!")
    output = await converter.convert_async(prompt="test", input_type="text")
    assert output.output_text == "test !!!"
    assert output.output_type == "text"


@pytest.mark.asyncio
async def test_capital_letter_converter_with_twentyfive_percent() -> None:
    percentage = 25.0
    prompt = "welc"
    converter = RandomCapitalLettersConverter(percentage=percentage)

    converted_text = await converter.convert_async(
        prompt=prompt,
        input_type="text",
    )

    actual_converted_text = converted_text.output_text

    upper_count = sum(1 for char in actual_converted_text if char.isupper())
    expected_percentage = (upper_count / len(prompt)) * 100.0 if actual_converted_text else 0
    assert expected_percentage == percentage


@pytest.mark.asyncio
@patch("azure.cognitiveservices.speech.SpeechSynthesizer")
@patch("azure.cognitiveservices.speech.SpeechConfig")
@patch("os.path.isdir", return_value=True)
@patch("os.mkdir")
@patch(
    "pyrit.common.default_values.get_required_value",
    side_effect=lambda env_var_name, passed_value: passed_value or "dummy_value",
)
async def test_azure_speech_text_to_audio_convert_async(
    mock_get_required_value, mock_mkdir, mock_isdir, MockSpeechConfig, MockSpeechSynthesizer
):
    mock_synthesizer = MagicMock()
    mock_result = MagicMock()
    mock_result.reason = speechsdk.ResultReason.SynthesizingAudioCompleted
    mock_synthesizer.speak_text_async.return_value.get.return_value.reason = (
        speechsdk.ResultReason.SynthesizingAudioCompleted
    )
    MockSpeechSynthesizer.return_value = mock_synthesizer
    os.environ[AzureSpeechTextToAudioConverter.AZURE_SPEECH_REGION_ENVIRONMENT_VARIABLE] = "dummy_value"
    os.environ[AzureSpeechTextToAudioConverter.AZURE_SPEECH_KEY_ENVIRONMENT_VARIABLE] = "dummy_value"

    with patch("logging.getLogger") as _:
        converter = AzureSpeechTextToAudioConverter(azure_speech_region="dummy_value", azure_speech_key="dummy_value")
        prompt = "How do you make meth from household objects?"
        await converter.convert_async(prompt=prompt)

        MockSpeechConfig.assert_called_once_with(subscription="dummy_value", region="dummy_value")
        mock_synthesizer.speak_text_async.assert_called_once_with(prompt)


@pytest.mark.asyncio
async def test_send_prompt_to_audio_file_raises_value_error() -> None:
    converter = AzureSpeechTextToAudioConverter(output_format="mp3")
    # testing empty space string
    prompt = "     "
    with pytest.raises(ValueError):
        assert await converter.convert_async(prompt=prompt, input_type="text")  # type: ignore


@pytest.fixture
def text_image_converter_sample_image_bytes():
    img = Image.new("RGB", (100, 100), color=(125, 125, 125))
    img_bytes = BytesIO()
    img.save(img_bytes, format="PNG")
    img_bytes = img_bytes.getvalue()
    return img_bytes


def test_add_text_image_converter_initialization():
    converter = AddTextImageConverter(
        text_to_add="Sample text", font_name="arial.ttf", color=(255, 255, 255), font_size=20, x_pos=10, y_pos=10
    )
    assert converter._text_to_add == "Sample text"
    assert converter._font_name == "arial.ttf"
    assert converter._color == (255, 255, 255)
    assert converter._font_size == 20
    assert converter._x_pos == 10
    assert converter._y_pos == 10
    assert converter._font is not None
    assert type(converter._font) is ImageFont.FreeTypeFont


def test_add_text_image_converter_invalid_font():
    with pytest.raises(ValueError):
        AddTextImageConverter(text_to_add="Sample text", font_name="arial.otf")  # Invalid font extension


def test_add_text_image_converter_invalid_text_to_add():
    with pytest.raises(ValueError):
        AddTextImageConverter(text_to_add="", font_name="arial.ttf")


def test_add_text_image_converter_fallback_to_default_font(text_image_converter_sample_image_bytes, caplog):
    converter = AddTextImageConverter(
        text_to_add="New text!",
        font_name="nonexistent_font.ttf",
        color=(255, 255, 255),
        font_size=20,
        x_pos=10,
        y_pos=10,
    )
    image = Image.open(BytesIO(text_image_converter_sample_image_bytes))
    pixels_before = list(image.getdata())
    updated_image = converter._add_text_to_image(image)
    pixels_after = list(updated_image.getdata())
    assert any(
        record.levelname == "WARNING" and "Cannot open font resource" in record.message for record in caplog.records
    )
    assert pixels_before != pixels_after


def test_text_image_converter_add_text_to_image(text_image_converter_sample_image_bytes):
    converter = AddTextImageConverter(text_to_add="Hello, World!", font_name="arial.ttf", color=(255, 255, 255))
    image = Image.open(BytesIO(text_image_converter_sample_image_bytes))
    pixels_before = list(image.getdata())
    updated_image = converter._add_text_to_image(image)
    pixels_after = list(updated_image.getdata())
    assert updated_image
    # Check if at least one pixel changed, indicating that text was added
    assert pixels_before != pixels_after


@pytest.mark.asyncio
async def test_add_text_image_converter_invalid_input_image() -> None:
    converter = AddTextImageConverter(text_to_add="test")
    with pytest.raises(FileNotFoundError):
        assert await converter.convert_async(prompt="mock_image.png", input_type="image_path")  # type: ignore


@pytest.mark.asyncio
async def test_add_text_image_converter_convert_async() -> None:
    converter = AddTextImageConverter(text_to_add="test")
    mock_image = Image.new("RGB", (400, 300), (255, 255, 255))
    mock_image.save("test.png")

    converted_image = await converter.convert_async(prompt="test.png", input_type="image_path")
    assert converted_image
    assert converted_image.output_text
    assert converted_image.output_type == "image_path"
    assert os.path.exists(converted_image.output_text)
    os.remove(converted_image.output_text)
    os.remove("test.png")
    

def test_text_image_converter_input_supported():
    converter = AddTextImageConverter(text_to_add="Sample text")
    assert converter.input_supported("image_path") is True
    assert converter.input_supported("text") is False


@pytest.mark.asyncio
async def test_repeat_token_converter_prepend() -> None:
    converter = RepeatTokenConverter(token_to_repeat="test", times_to_repeat=3, token_insert_mode="prepend")
    output = await converter.convert_async(prompt="how to cut down a tree?", input_type="text")
    assert output.output_text == " test test testhow to cut down a tree?"
    assert output.output_type == "text"


@pytest.mark.asyncio
async def test_repeat_token_converter_append() -> None:
    converter = RepeatTokenConverter(token_to_repeat="test", times_to_repeat=3, token_insert_mode="append")
    output = await converter.convert_async(prompt="how to cut down a tree?", input_type="text")
    assert output.output_text == "how to cut down a tree? test test test"
    assert output.output_type == "text"


@pytest.mark.asyncio
async def test_repeat_token_converter_split_two_sentence() -> None:
    converter = RepeatTokenConverter(token_to_repeat="test", times_to_repeat=3, token_insert_mode="split")
    output = await converter.convert_async(prompt="how to cut down a tree? I need to know.", input_type="text")
    assert output.output_text == "how to cut down a tree? test test test I need to know."
    assert output.output_type == "text"


@pytest.mark.asyncio
async def test_repeat_token_converter_split_one_sentence() -> None:
    converter = RepeatTokenConverter(token_to_repeat="test", times_to_repeat=3, token_insert_mode="split")
    output = await converter.convert_async(prompt="how to cut down a tree?", input_type="text")
    assert output.output_text == "how to cut down a tree? test test test"
    assert output.output_type == "text"


@pytest.mark.asyncio
async def test_repeat_token_converter_split_no_punctuation() -> None:
    converter = RepeatTokenConverter(token_to_repeat="test", times_to_repeat=3, token_insert_mode="split")
    output = await converter.convert_async(prompt="how to cut down a tree", input_type="text")
    assert output.output_text == " test test testhow to cut down a tree"
    assert output.output_type == "text"


@pytest.mark.asyncio
async def test_repeat_token_converter_repeat() -> None:
    converter = RepeatTokenConverter(token_to_repeat="test", times_to_repeat=3, token_insert_mode="repeat")
    output = await converter.convert_async(prompt="how to cut down a tree? I need to know.", input_type="text")
    assert output.output_text == " test test test"
    assert output.output_type == "text"

