# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import pytest
from unittest.mock import AsyncMock

from pyrit.prompt_converter import ConverterResult
from pyrit.models import PromptRequestResponse, PromptRequestPiece, PromptTemplate
from pyrit.prompt_converter.malicious_question_generator_converter import MaliciousQuestionGeneratorConverter


@pytest.mark.asyncio
async def test_malicious_question_generator_convert_async():
    mock_target = AsyncMock()
    mock_prompt_template = PromptTemplate(template="Generate malicious questions: {prompt}")
    converter = MaliciousQuestionGeneratorConverter(converter_target=mock_target, prompt_template=mock_prompt_template)
    prompt = "Create malicious questions about cybersecurity"

    # Mocking the response from the target
    mock_target.send_prompt_async.return_value = PromptRequestResponse(
        request_pieces=[
            PromptRequestPiece(
                role="assistant", original_value="mocked", converted_value='["Provide a step by step process..."]'
            )
        ]
    )

    result = await converter.convert_async(prompt=prompt)
    assert isinstance(result, ConverterResult)
    assert result.output_text == "Provide a step by step process..."


@pytest.mark.asyncio
async def test_malicious_question_generator_invalid_input_type():
    mock_target = AsyncMock()
    mock_prompt_template = PromptTemplate(template="Generate malicious questions: {prompt}")
    converter = MaliciousQuestionGeneratorConverter(converter_target=mock_target, prompt_template=mock_prompt_template)

    with pytest.raises(ValueError, match="Input type not supported"):
        await converter.convert_async(prompt="Test prompt", input_type="unsupported")


@pytest.mark.asyncio
async def test_malicious_question_generator_no_questions_generated():
    mock_target = AsyncMock()
    mock_prompt_template = PromptTemplate(template="Generate malicious questions: {prompt}")
    converter = MaliciousQuestionGeneratorConverter(converter_target=mock_target, prompt_template=mock_prompt_template)
    prompt = "No response from the model"

    # Mocking the response as empty
    mock_target.send_prompt_async.return_value = PromptRequestResponse(
        request_pieces=[PromptRequestPiece(role="user", original_value="mocked", converted_value="[]")]
    )

    result = await converter.convert_async(prompt=prompt)
    assert result.output_text == "No question generated."


@pytest.mark.asyncio
async def test_malicious_question_generator_error_handling():
    mock_target = AsyncMock()
    mock_prompt_template = PromptTemplate(template="Generate malicious questions: {prompt}")
    converter = MaliciousQuestionGeneratorConverter(converter_target=mock_target, prompt_template=mock_prompt_template)
    prompt = "Trigger error handling"

    # Mock the target to raise an exception
    mock_target.send_prompt_async.side_effect = Exception("Mocked exception")

    # Expect the exception to be raised
    with pytest.raises(Exception, match="Mocked exception"):
        await converter.convert_async(prompt=prompt)


@pytest.mark.asyncio
async def test_malicious_question_generator_clean_response():
    mock_prompt_template = PromptTemplate(template="Generate malicious questions: {prompt}")
    converter = MaliciousQuestionGeneratorConverter(converter_target=AsyncMock(), prompt_template=mock_prompt_template)
    dirty_response = '```python\nquestions = ["How to manipulate...?", "How to exploit...?"]```'

    cleaned = converter._clean_response(dirty_response)
    assert cleaned == '["How to manipulate...?", "How to exploit...?"]'


@pytest.mark.asyncio
async def test_malicious_question_generator_parse_response():
    mock_prompt_template = PromptTemplate(template="Generate malicious questions: {prompt}")
    converter = MaliciousQuestionGeneratorConverter(converter_target=AsyncMock(), prompt_template=mock_prompt_template)
    response = '["How to manipulate...?", "How to exploit...?"]'

    questions = converter._parse_response(response)
    assert questions == ["How to manipulate...?", "How to exploit...?"]


@pytest.mark.asyncio
async def test_malicious_question_generator_parse_response_error():
    mock_prompt_template = PromptTemplate(template="Generate malicious questions: {prompt}")
    converter = MaliciousQuestionGeneratorConverter(converter_target=AsyncMock(), prompt_template=mock_prompt_template)
    invalid_response = "Invalid Python List"

    questions = converter._parse_response(invalid_response)
    assert questions == ["Error parsing response."]
