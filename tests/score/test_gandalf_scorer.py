# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import uuid
from typing import Generator, Optional
from unittest.mock import AsyncMock, MagicMock, patch
import pytest

from pyrit.exceptions.exception_classes import PyritException
from pyrit.memory import CentralMemory
from pyrit.memory.memory_interface import MemoryInterface
from pyrit.models import PromptRequestPiece
from pyrit.models import PromptRequestResponse
from pyrit.score import GandalfScorer
from pyrit.prompt_target import GandalfLevel
from tests.mocks import get_memory_interface


def generate_password_extraction_response(
    response_text: str, conversation_id: Optional[str] = None
) -> PromptRequestResponse:
    return PromptRequestResponse(
        request_pieces=[
            PromptRequestPiece(
                role="assistant",
                original_value=response_text,
                original_value_data_type="text",
                converted_value=response_text,
                converted_value_data_type="text",
                conversation_id=conversation_id or str(uuid.uuid4()),
            )
        ]
    )


def generate_request(conversation_id: Optional[str] = None) -> PromptRequestResponse:
    return PromptRequestResponse(
        request_pieces=[
            PromptRequestPiece(
                role="user",
                original_value="some request",
                original_value_data_type="text",
                converted_value="some request",
                converted_value_data_type="text",
                conversation_id=conversation_id or str(uuid.uuid4()),
            )
        ]
    )


@pytest.fixture
def memory() -> Generator[MemoryInterface, None, None]:
    yield from get_memory_interface()


@patch("requests.post")
@pytest.mark.parametrize("password_correct", [True, False])
@pytest.mark.parametrize("level", [GandalfLevel.LEVEL_1, GandalfLevel.LEVEL_2, GandalfLevel.LEVEL_3])
@pytest.mark.asyncio
async def test_gandalf_scorer_score(mocked_post, memory: MemoryInterface, level: GandalfLevel, password_correct: bool):
    chat_target = MagicMock()

    conversation_id = str(uuid.uuid4())
    memory.add_request_response_to_memory(request=generate_request(conversation_id=conversation_id))
    response = generate_password_extraction_response("SUNSHINE", conversation_id=conversation_id)
    memory.add_request_response_to_memory(request=response)

    chat_target.send_prompt_async = AsyncMock(return_value=response)
    with patch.object(CentralMemory, "get_memory_instance", return_value=memory):

        scorer = GandalfScorer(level=level, chat_target=chat_target)

        mocked_post.return_value = MagicMock(
            status_code=200, json=lambda: {"success": password_correct, "message": "Message"}
        )

        scores = await scorer.score_async(response.request_pieces[0])

        assert len(scores) == 1
        assert scores[0].get_value() == password_correct
        assert (
            scores[0].score_value_description
            == "When password is found in text, the score is True. The rationale will contain the password."
        )
        if password_correct:
            assert scores[0].score_rationale == "Password SUNSHINE found! Gandalf response: Message"
        else:
            assert scores[0].score_rationale == "Invalid password found in text. [value=SUNSHINE]"
        assert scores[0].scorer_class_identifier["__type__"] == "GandalfScorer"


@patch("requests.post")
@pytest.mark.parametrize("level", [GandalfLevel.LEVEL_1, GandalfLevel.LEVEL_2, GandalfLevel.LEVEL_3])
@pytest.mark.asyncio
async def test_gandalf_scorer_set_system_prompt(
    mocked_post,
    memory: MemoryInterface,
    level: GandalfLevel,
):
    conversation_id = str(uuid.uuid4())
    memory.add_request_response_to_memory(request=generate_request(conversation_id=conversation_id))
    response = generate_password_extraction_response("SUNSHINE", conversation_id=conversation_id)
    memory.add_request_response_to_memory(request=response)

    chat_target = MagicMock()
    chat_target.send_prompt_async = AsyncMock(return_value=response)

    with patch.object(CentralMemory, "get_memory_instance", return_value=memory):
        scorer = GandalfScorer(chat_target=chat_target, level=level)

        mocked_post.return_value = MagicMock(status_code=200, json=lambda: {"success": True, "message": "Message"})

        await scorer.score_async(response.request_pieces[0])

        chat_target.set_system_prompt.assert_called_once()

        mocked_post.assert_called_once()


@pytest.mark.parametrize("level", [GandalfLevel.LEVEL_1, GandalfLevel.LEVEL_2, GandalfLevel.LEVEL_3])
@pytest.mark.asyncio
async def test_gandalf_scorer_adds_to_memory(level: GandalfLevel, memory: MemoryInterface):
    conversation_id = str(uuid.uuid4())
    generated_request = generate_request(conversation_id=conversation_id)
    memory.add_request_response_to_memory(request=generated_request)
    response = generate_password_extraction_response("SUNSHINE", conversation_id=conversation_id)
    memory.add_request_response_to_memory(request=response)

    chat_target = MagicMock()
    chat_target.send_prompt_async = AsyncMock(return_value=response)
    with patch.object(CentralMemory, "get_memory_instance", return_value=memory):
        with patch.object(
            memory, "get_prompt_request_pieces_by_id", return_value=[generated_request.request_pieces[0]]
        ):
            scorer = GandalfScorer(level=level, chat_target=chat_target)

            await scorer.score_async(response.request_pieces[0])


@pytest.mark.parametrize("level", [GandalfLevel.LEVEL_1, GandalfLevel.LEVEL_2, GandalfLevel.LEVEL_3])
@pytest.mark.asyncio
async def test_gandalf_scorer_runtime_error_retries(level: GandalfLevel, memory: MemoryInterface):

    conversation_id = str(uuid.uuid4())
    memory.add_request_response_to_memory(request=generate_request(conversation_id=conversation_id))
    response = generate_password_extraction_response("SUNSHINE", conversation_id=conversation_id)
    memory.add_request_response_to_memory(request=response)

    chat_target = MagicMock()
    chat_target.send_prompt_async = AsyncMock(side_effect=[RuntimeError("Error"), response])
    with patch.object(CentralMemory, "get_memory_instance", return_value=memory):
        scorer = GandalfScorer(level=level, chat_target=chat_target)

        with pytest.raises(PyritException):
            await scorer.score_async(response.request_pieces[0])

        assert chat_target.send_prompt_async.call_count == 1
