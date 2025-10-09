# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import MutableSequence
from unittest.mock import MagicMock, patch

import pytest
from unit.mocks import get_audio_request_piece, get_sample_conversations

from pyrit.memory.central_memory import CentralMemory
from pyrit.models import PromptRequestPiece, PromptRequestResponse
from pyrit.prompt_target import PromptShieldTarget


@pytest.fixture
def audio_request_piece() -> PromptRequestPiece:
    return get_audio_request_piece()


@pytest.fixture
def sample_conversations() -> MutableSequence[PromptRequestPiece]:
    conversations = get_sample_conversations()
    return PromptRequestResponse.flatten_to_prompt_request_pieces(conversations)


@pytest.fixture
def promptshield_target(sqlite_instance) -> PromptShieldTarget:
    return PromptShieldTarget(endpoint="mock", api_key="mock")


@pytest.fixture
def promptshield_target_with_entra(sqlite_instance):
    target = PromptShieldTarget(endpoint="https://test.endpoint.com", use_entra_auth=True)
    target._azure_auth = MagicMock()
    target._azure_auth.refresh_token = MagicMock(return_value="test_access_token")
    return target


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
async def test_prompt_shield_validate_request_length(promptshield_target: PromptShieldTarget):
    request = PromptRequestResponse(
        request_pieces=[
            PromptRequestPiece(role="user", conversation_id="123", original_value="test1"),
            PromptRequestPiece(role="user", conversation_id="123", original_value="test2"),
        ]
    )
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


def test_use_entra_auth_true_with_api_key_raises_error(sqlite_instance):
    """Test that use_entra_auth=True with api_key raises ValueError."""
    with pytest.raises(ValueError, match="If using Entra ID auth, please do not specify api_key."):
        with patch.object(CentralMemory, "get_memory_instance", return_value=sqlite_instance):
            PromptShieldTarget(
                endpoint="https://test.endpoint.com",
                api_key="test_key",
                use_entra_auth=True,
            )


def test_use_entra_auth_true_uses_credential(promptshield_target_with_entra: PromptShieldTarget):
    """Test that use_entra_auth=True uses Azure authentication."""
    # Verify the target was created with Entra auth
    assert promptshield_target_with_entra is not None
    assert promptshield_target_with_entra._azure_auth is not None
    assert promptshield_target_with_entra._azure_auth.refresh_token() == "test_access_token"


def test_use_entra_auth_false_uses_api_key():
    """Test that use_entra_auth=False uses API key authentication."""
    target = PromptShieldTarget(endpoint="https://test.endpoint.com", api_key="test_key", use_entra_auth=False)

    # Verify target was created successfully with API key
    assert target is not None
    assert target._api_key == "test_key"


def test_use_entra_auth_default_false_uses_api_key():
    """Test that default behavior (use_entra_auth=False) uses API key authentication."""
    target = PromptShieldTarget(
        endpoint="https://test.endpoint.com",
        api_key="test_key",
        # use_entra_auth not specified, should default to False
    )

    # Verify target was created successfully with API key (not Entra auth)
    assert target is not None
    assert target._api_key == "test_key"
