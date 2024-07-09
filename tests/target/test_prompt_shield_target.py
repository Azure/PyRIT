# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import pytest
from unittest.mock import Mock
from tests.mocks import get_sample_conversations, get_audio_request_piece

from pyrit.models import PromptRequestPiece, PromptRequestResponse
from pyrit.prompt_target import PromptShieldTarget

@pytest.fixture
def audio_request_piece() -> PromptRequestPiece:
    return get_audio_request_piece()

@pytest.fixture
def sample_conversations() -> list[PromptRequestPiece]:
    return get_sample_conversations()

@pytest.fixture
def promptshield_target() -> PromptShieldTarget:
    return PromptShieldTarget(
        endpoint="mock",
        api_key="mock"
    )

@pytest.fixture
def sample_delineated_prompt_as_str() -> str:
    sample: str = '''
    Mock userPrompt
    <document>
    mock document
    </document>
    '''
    return sample

@pytest.fixture
def sample_delineated_prompt_as_dict() -> dict:
    sample: dict = {'userPrompt': '\n    Mock userPrompt\n    ', 'documents': ['\n    mock document\n    ']}
    return sample

@pytest.fixture
def sample_conversation_piece() -> PromptRequestPiece:
    prp = PromptRequestPiece(
        role='user',
        original_value=sample_delineated_prompt_as_str
    )
    return prp

@pytest.fixture
def sample_conversations() -> list[PromptRequestPiece]:
    return get_sample_conversations()

def test_promptshield_init(promptshield_target: PromptShieldTarget):
    assert promptshield_target

@pytest.mark.asyncio
async def test_prompt_shield_validate_request_length(
        promptshield_target: PromptShieldTarget,
        sample_conversations: list[PromptRequestPiece]
):
    request = PromptRequestResponse(request_pieces=sample_conversations)
    with pytest.raises(
        ValueError, 
        match="Sorry, but requests with multiple entries are not supported. " \
              "Please wrap each PromptRequestPiece in a PromptRequestResponse." ):
        
        await promptshield_target.send_prompt_async(prompt_request=request)

@pytest.mark.asyncio
async def test_prompt_shield_reject_non_text(
    promptshield_target: PromptShieldTarget,
    audio_request_piece: PromptRequestPiece
):
    with pytest.raises(ValueError):
        await promptshield_target.send_prompt_async(
            prompt_request=PromptRequestResponse([audio_request_piece])
        )

@pytest.mark.asyncio
async def test_prompt_shield_document_parsing(
    promptshield_target: PromptShieldTarget,
    sample_delineated_prompt_as_str: str,
    sample_delineated_prompt_as_dict
):
    result = promptshield_target._input_parser(sample_delineated_prompt_as_str)

    print("RESULT:")
    print(result)
    print("SAMPLE:")
    print(sample_delineated_prompt_as_dict)

    assert result == sample_delineated_prompt_as_dict

    

# Add: empty field(s) (edge case)
