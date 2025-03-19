# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import json
from typing import MutableSequence
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from unit.mocks import get_sample_conversations, openai_response_json_dict

from pyrit.models import PromptRequestPiece, PromptRequestResponse
from pyrit.orchestrator.orchestrator_class import Orchestrator
from pyrit.prompt_target import OpenAIChatTarget


@pytest.fixture
def sample_entries() -> MutableSequence[PromptRequestPiece]:
    return get_sample_conversations()


@pytest.fixture
def openai_response_json() -> dict:
    return openai_response_json_dict()


@pytest.fixture
def azure_openai_target(patch_central_database):
    return OpenAIChatTarget(
        endpoint="test",
        api_key="test",
    )


def test_set_system_prompt(azure_openai_target: OpenAIChatTarget):
    azure_openai_target.set_system_prompt(
        system_prompt="system prompt",
        conversation_id="1",
        orchestrator_identifier=Orchestrator().get_identifier(),
        labels={},
    )

    chats = azure_openai_target._memory.get_prompt_request_pieces(conversation_id="1")
    assert len(chats) == 1, f"Expected 1 chat, got {len(chats)}"
    assert chats[0].role == "system"
    assert chats[0].converted_value == "system prompt"


@pytest.mark.asyncio
@pytest.mark.asyncio
async def test_set_system_prompt_adds_memory(azure_openai_target: OpenAIChatTarget):
    azure_openai_target.set_system_prompt(
        system_prompt="system prompt",
        conversation_id="1",
        orchestrator_identifier=Orchestrator().get_identifier(),
        labels={},
    )

    chats = azure_openai_target._memory.get_prompt_request_pieces(conversation_id="1")
    assert len(chats) == 1, f"Expected 1 chats, got {len(chats)}"
    assert chats[0].role == "system"


@pytest.mark.asyncio
async def test_send_prompt_with_system_calls_chat_complete(
    azure_openai_target: OpenAIChatTarget,
    openai_response_json: dict,
    sample_entries: MutableSequence[PromptRequestPiece],
):

    openai_mock_return = MagicMock()
    openai_mock_return.text = json.dumps(openai_response_json)

    with patch("pyrit.common.net_utility.make_request_and_raise_if_error_async", new_callable=AsyncMock) as mock_create:

        mock_create.return_value = openai_mock_return

        azure_openai_target.set_system_prompt(
            system_prompt="system prompt",
            conversation_id="1",
            orchestrator_identifier=Orchestrator().get_identifier(),
            labels={},
        )

        request = sample_entries[0]
        request.converted_value = "hi, I am a victim chatbot, how can I help?"
        request.conversation_id = "1"

        await azure_openai_target.send_prompt_async(prompt_request=PromptRequestResponse(request_pieces=[request]))

        mock_create.assert_called_once()


@pytest.mark.asyncio
async def test_send_prompt_async_with_delay(
    azure_openai_target: OpenAIChatTarget,
    openai_response_json: dict,
    sample_entries: MutableSequence[PromptRequestPiece],
):
    azure_openai_target._max_requests_per_minute = 10

    openai_mock_return = MagicMock()
    openai_mock_return.text = json.dumps(openai_response_json)

    with patch("pyrit.common.net_utility.make_request_and_raise_if_error_async", new_callable=AsyncMock) as mock_create:
        with patch("asyncio.sleep") as mock_sleep:
            mock_create.return_value = openai_mock_return

            request = sample_entries[0]
            request.converted_value = "hi, I am a victim chatbot, how can I help?"

            await azure_openai_target.send_prompt_async(prompt_request=PromptRequestResponse(request_pieces=[request]))

            mock_create.assert_called_once()
            mock_sleep.assert_called_once_with(6)  # 60/max_requests_per_minute
