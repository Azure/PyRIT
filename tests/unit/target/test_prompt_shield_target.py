# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import MutableSequence

import pytest
from unit.mocks import get_audio_request_piece, get_sample_conversations

from pyrit.models import PromptRequestPiece, PromptRequestResponse
from pyrit.prompt_target import PromptShieldTarget


@pytest.fixture
def audio_request_piece() -> PromptRequestPiece:
    return get_audio_request_piece()


@pytest.fixture
def sample_conversations() -> MutableSequence[PromptRequestPiece]:
    return get_sample_conversations()


@pytest.fixture
def promptshield_target() -> PromptShieldTarget:
    return PromptShieldTarget(endpoint="mock", api_key="mock")


@pytest.fixture
def sample_delineated_prompt_as_str() -> str:
    sample: str = """
    Mock userPrompt
    <document>
    mock document
    </document>
    """
    return sample


@pytest.fixture
def sample_delineated_prompt_as_dict() -> dict:
    sample: dict = {"userPrompt": "\n    Mock userPrompt\n    ", "documents": ["\n    mock document\n    "]}
    return sample


@pytest.fixture
def sample_conversation_piece(sample_delineated_prompt_as_str: str) -> PromptRequestPiece:
    prp = PromptRequestPiece(role="user", original_value=sample_delineated_prompt_as_str)
    return prp


def test_promptshield_init(promptshield_target: PromptShieldTarget):
    assert promptshield_target


@pytest.mark.asyncio
async def test_prompt_shield_validate_request_length(
    promptshield_target: PromptShieldTarget, sample_conversations: MutableSequence[PromptRequestPiece]
):
    request = PromptRequestResponse(request_pieces=sample_conversations)
    with pytest.raises(ValueError, match="This target only supports a single prompt request piece."):

        await promptshield_target.send_prompt_async(prompt_request=request)


@pytest.mark.asyncio
async def test_prompt_shield_reject_non_text(
    promptshield_target: PromptShieldTarget, audio_request_piece: PromptRequestPiece
):
    with pytest.raises(ValueError):
        await promptshield_target.send_prompt_async(prompt_request=PromptRequestResponse([audio_request_piece]))


@pytest.mark.asyncio
async def test_prompt_shield_document_parsing(
    promptshield_target: PromptShieldTarget, sample_delineated_prompt_as_str: str, sample_delineated_prompt_as_dict
):
    result = promptshield_target._input_parser(sample_delineated_prompt_as_str)

    assert result == sample_delineated_prompt_as_dict


@pytest.mark.asyncio
async def test_prompt_shield_response_validation(promptshield_target: PromptShieldTarget):
    # This tests handling both an empty request and an empty response
    promptshield_target._validate_response(request_body=dict(), response_body=dict())
