# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import MutableSequence
from unittest.mock import MagicMock, patch

import pytest
from unit.mocks import get_audio_request_piece, get_sample_conversations

from pyrit.memory.central_memory import CentralMemory
from pyrit.models import MessagePiece, Message
from pyrit.prompt_target import PromptShieldTarget


@pytest.fixture
def audio_request_piece() -> MessagePiece:
    return get_audio_request_piece()


@pytest.fixture
def sample_conversations() -> MutableSequence[MessagePiece]:
    conversations = get_sample_conversations()
    return Message.flatten_to_prompt_request_pieces(conversations)


@pytest.fixture
def promptshield_target(sqlite_instance) -> PromptShieldTarget:
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
def sample_conversation_piece(sample_delineated_prompt_as_str: str) -> MessagePiece:
    prp = MessagePiece(role="user", original_value=sample_delineated_prompt_as_str)
    return prp


def test_promptshield_init(promptshield_target: PromptShieldTarget):
    assert promptshield_target


@pytest.mark.asyncio
async def test_prompt_shield_validate_request_length(promptshield_target: PromptShieldTarget):
    request = Message(
        request_pieces=[
            MessagePiece(role="user", conversation_id="123", original_value="test1"),
            MessagePiece(role="user", conversation_id="123", original_value="test2"),
        ]
    )
    with pytest.raises(ValueError, match="This target only supports a single prompt request piece."):
        await promptshield_target.send_prompt_async(prompt_request=request)


@pytest.mark.asyncio
async def test_prompt_shield_reject_non_text(
    promptshield_target: PromptShieldTarget, audio_request_piece: MessagePiece
):
    with pytest.raises(ValueError):
        await promptshield_target.send_prompt_async(prompt_request=Message([audio_request_piece]))


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


def test_use_entra_auth_true_uses_credential(sqlite_instance):
    """Test that use_entra_auth=True uses Azure authentication."""
    with (
        patch("pyrit.prompt_target.prompt_shield_target.get_default_scope") as mock_scope,
        patch("pyrit.prompt_target.prompt_shield_target.AzureAuth") as mock_auth_class,
    ):

        mock_scope.return_value = "https://cognitiveservices.azure.com/.default"
        mock_auth_instance = MagicMock()
        mock_auth_class.return_value = mock_auth_instance

        target = PromptShieldTarget(endpoint="https://test.endpoint.com", use_entra_auth=True)

        # Verify Azure Auth was used correctly
        mock_scope.assert_called_once_with("https://test.endpoint.com")
        mock_auth_class.assert_called_once_with(token_scope="https://cognitiveservices.azure.com/.default")

        # Verify target was created successfully with Entra auth
        assert target is not None
        assert target._azure_auth == mock_auth_instance
        assert target._api_key is None


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
