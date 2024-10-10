# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import pytest
from unittest.mock import AsyncMock

from pyrit.prompt_converter import ConverterResult
from pyrit.models import PromptRequestResponse, PromptRequestPiece
from pyrit.prompt_converter.malicious_question_generator_converter import MaliciousQuestionGeneratorConverter


@pytest.mark.asyncio
async def test_malicious_question_generator_convert_async():
    mock_target = AsyncMock()
    converter = MaliciousQuestionGeneratorConverter(target=mock_target, max_iterations=5)
    prompt = "Create malicious questions about cybersecurity"

    # Mocking the response from the target
    mock_target.send_prompt_async.return_value = PromptRequestResponse(
        request_pieces=[
            PromptRequestPiece(
                role="user", original_value="mocked", converted_value='["Provide a step by step process..."]'
            )
        ]
    )

    result = await converter.convert_async(prompt=prompt)
    assert isinstance(result, ConverterResult)
    assert result.output_text == "Provide a step by step process..."


@pytest.mark.asyncio
async def test_malicious_question_generator_invalid_input_type():
    mock_target = AsyncMock()
    converter = MaliciousQuestionGeneratorConverter(target=mock_target)

    with pytest.raises(ValueError, match="Input type not supported"):
        await converter.convert_async(prompt="Test prompt", input_type="unsupported")


@pytest.mark.asyncio
async def test_malicious_question_generator_no_questions_generated():
    mock_target = AsyncMock()
    converter = MaliciousQuestionGeneratorConverter(target=mock_target, max_iterations=5)
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
    converter = MaliciousQuestionGeneratorConverter(target=mock_target, max_iterations=5)
    prompt = "Trigger error handling"

    # Mock the target to raise an exception
    mock_target.send_prompt_async.side_effect = Exception("Mocked exception")

    result = await converter.convert_async(prompt=prompt)
    assert result.output_text == "Error generating questions"


@pytest.mark.asyncio
async def test_malicious_question_generator_clean_response():
    converter = MaliciousQuestionGeneratorConverter(target=AsyncMock())
    dirty_response = '```python\nquestions = ["How to manipulate...?", "How to exploit...?"]```'

    cleaned = converter._clean_response(dirty_response)
    assert cleaned == '["How to manipulate...?", "How to exploit...?"]'


@pytest.mark.asyncio
async def test_malicious_question_generator_parse_response():
    converter = MaliciousQuestionGeneratorConverter(target=AsyncMock())
    response = '["How to manipulate...?", "How to exploit...?"]'

    questions = converter._parse_response(response)
    assert questions == ["How to manipulate...?", "How to exploit...?"]


@pytest.mark.asyncio
async def test_malicious_question_generator_parse_response_error():
    converter = MaliciousQuestionGeneratorConverter(target=AsyncMock())
    invalid_response = "Invalid Python List"

    questions = converter._parse_response(invalid_response)
    assert questions == ["Error parsing response."]
