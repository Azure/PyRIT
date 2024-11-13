# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Generator
import uuid
from unittest.mock import Mock, AsyncMock, ANY
from unittest.mock import patch

import pytest

from pyrit.memory import MemoryInterface
from pyrit.models import PromptRequestResponse, Score
from pyrit.orchestrator import PAIROrchestrator
from pyrit.orchestrator.pair_orchestrator import PromptRequestPiece
from pyrit.prompt_target import OpenAIChatTarget
from pyrit.score import Scorer
from pyrit.memory import CentralMemory
from tests.mocks import get_memory_interface


def _build_prompt_response_with_single_prompt_piece(*, prompt: str) -> PromptRequestResponse:
    return PromptRequestResponse(
        request_pieces=[PromptRequestPiece(original_value=prompt, converted_value=prompt, role="user")]
    )


@pytest.fixture
def memory_interface() -> Generator[MemoryInterface, None, None]:
    yield from get_memory_interface()


@pytest.fixture
def mock_central_memory_instance(memory_interface):
    """Fixture to mock CentralMemory.get_memory_instance"""
    with patch.object(CentralMemory, "get_memory_instance", return_value=memory_interface) as duck_db_memory:
        yield duck_db_memory


@pytest.fixture
def chat_completion_engine() -> OpenAIChatTarget:
    return OpenAIChatTarget(deployment_name="test", endpoint="test", api_key="test")


@pytest.fixture
def scorer_mock() -> Scorer:
    scorer = Mock()
    scorer.scorer_type = "float_scale"
    scorer.score_async = AsyncMock(return_value=[])
    return scorer


@pytest.fixture
def orchestrator(mock_central_memory_instance: MemoryInterface, scorer_mock: Scorer) -> PAIROrchestrator:
    target = Mock()
    attacker = Mock()
    labels = {"op_name": "name1"}
    orchestrator = PAIROrchestrator(
        prompt_target=target,
        desired_target_response_prefix="desired response",
        red_teaming_chat=attacker,
        conversation_objective="attacker objective",
        memory_labels=labels,
        scorer=scorer_mock,
        stop_on_first_success=True,
        number_of_conversation_streams=3,
        max_conversation_depth=5,
    )
    return orchestrator


@pytest.fixture
def correctly_formatted_response_piece() -> PromptRequestPiece:
    return PromptRequestPiece(
        original_value='{"prompt": "prompt", "improvement": "improvement"}',
        converted_value='{"prompt": "prompt", "improvement": "improvement"}',
        role="user",
    )


@pytest.mark.asyncio
async def test_init(orchestrator):
    assert orchestrator._prompt_target is not None
    assert orchestrator._adversarial_target is not None
    assert orchestrator._scorer is not None
    assert orchestrator._conversation_objective == "attacker objective"
    assert orchestrator._desired_target_response_prefix == "desired response"
    assert orchestrator._global_memory_labels == {"op_name": "name1"}


@pytest.mark.asyncio
async def test_run(orchestrator: PAIROrchestrator):
    NUM_CONVERSATIONS_STREAMS = 3
    orchestrator._number_of_conversation_streams = NUM_CONVERSATIONS_STREAMS
    orchestrator._process_conversation_stream = AsyncMock(return_value=[])  # type: ignore
    orchestrator._should_stop = Mock(return_value=False)  # type: ignore
    result = await orchestrator.run()
    assert result == []
    orchestrator._process_conversation_stream.call_count == NUM_CONVERSATIONS_STREAMS
    orchestrator._should_stop.call_count == NUM_CONVERSATIONS_STREAMS


@pytest.mark.asyncio
async def test_output_is_properly_formatted_when_jailbreak_is_found(
    orchestrator: PAIROrchestrator,
):
    orchestrator._get_attacker_response_and_store = AsyncMock(  # type: ignore
        return_value=_build_prompt_response_with_single_prompt_piece(prompt='{"improvement": "aaaa", "prompt": "bbb"}')
    )
    orchestrator._get_target_response_and_store = AsyncMock(  # type: ignore
        return_value=PromptRequestResponse(
            request_pieces=[
                PromptRequestPiece(
                    original_value="aaaa",
                    converted_value="aaaa",
                    role="user",
                )
            ]
        )
    )
    orchestrator._scorer.score_async = AsyncMock(  # type: ignore
        return_value=[
            Score(
                score_value="1.0",
                score_value_description="",
                score_type="float_scale",
                score_category="",
                score_rationale="",
                score_metadata="",
                prompt_request_response_id="",
            )
        ]
    )
    result = await orchestrator.run()
    assert len(result) == 1


@pytest.mark.asyncio
async def test_output_is_properly_formatted_when_jailbreak_is_found_multiple_streams(
    orchestrator: PAIROrchestrator,
):
    NUM_CONVERSATION_STREAMS = 20
    orchestrator._number_of_conversation_streams = NUM_CONVERSATION_STREAMS
    orchestrator._stop_on_first_success = False
    orchestrator._get_attacker_response_and_store = AsyncMock(  # type: ignore
        return_value=_build_prompt_response_with_single_prompt_piece(prompt='{"improvement": "aaaa", "prompt": "bbb"}')
    )
    orchestrator._get_target_response_and_store = AsyncMock(  # type: ignore
        return_value=PromptRequestResponse(
            request_pieces=[
                PromptRequestPiece(
                    original_value="aaaa",
                    converted_value="aaaa",
                    role="user",
                )
            ]
        )
    )
    orchestrator._scorer.score_async = AsyncMock(  # type: ignore
        return_value=[
            Score(
                score_value="1.0",
                score_value_description="",
                score_type="float_scale",
                score_category="",
                score_rationale="",
                score_metadata="",
                prompt_request_response_id="",
            )
        ]
    )
    result = await orchestrator.run()
    assert len(result) == NUM_CONVERSATION_STREAMS


@pytest.mark.asyncio
async def test_output_is_properly_formatted_when_jailbreak_is_not_found(
    orchestrator: PAIROrchestrator,
):
    orchestrator._get_attacker_response_and_store = AsyncMock(  # type: ignore
        return_value=_build_prompt_response_with_single_prompt_piece(prompt='{"improvement": "aaaa", "prompt": "bbb"}')
    )
    orchestrator._get_target_response_and_store = AsyncMock(  # type: ignore
        return_value=PromptRequestResponse(
            request_pieces=[
                PromptRequestPiece(
                    original_value="aaaa",
                    converted_value="aaaa",
                    role="user",
                )
            ]
        )
    )
    orchestrator._scorer.score_async = AsyncMock(  # type: ignore
        return_value=[
            Score(
                score_value="0.0",
                score_value_description="",
                score_type="float_scale",
                score_category="",
                score_rationale="",
                score_metadata="",
                prompt_request_response_id="",
            )
        ]
    )
    result = await orchestrator.run()
    assert len(result) == 0


@pytest.mark.asyncio
async def test_correct_number_of_streams_executes(
    orchestrator: PAIROrchestrator,
):

    MAX_CONVERSATION_DEPTH = 5
    MAX_CONVERSATION_STREAMS = 10
    orchestrator._number_of_conversation_streams = MAX_CONVERSATION_STREAMS
    orchestrator._max_conversation_depth = MAX_CONVERSATION_DEPTH
    orchestrator._get_attacker_response_and_store = AsyncMock(  # type: ignore
        return_value=_build_prompt_response_with_single_prompt_piece(prompt='{"improvement": "aaaa", "prompt": "bbb"}')
    )
    orchestrator._get_target_response_and_store = AsyncMock(  # type: ignore
        return_value=PromptRequestResponse(
            request_pieces=[
                PromptRequestPiece(
                    original_value="aaaa",
                    converted_value="aaaa",
                    role="user",
                )
            ]
        )
    )
    orchestrator._scorer.score_async = AsyncMock(  # type: ignore
        return_value=[
            Score(
                score_value="0.0",
                score_value_description="",
                score_type="float_scale",
                score_category="",
                score_rationale="",
                score_metadata="",
                prompt_request_response_id="",
            )
        ]
    )
    result = await orchestrator.run()
    assert len(result) == 0
    assert orchestrator._scorer.score_async.call_count == MAX_CONVERSATION_DEPTH * MAX_CONVERSATION_STREAMS
    assert orchestrator._get_attacker_response_and_store.call_count == MAX_CONVERSATION_DEPTH * MAX_CONVERSATION_STREAMS
    assert orchestrator._get_target_response_and_store.call_count == MAX_CONVERSATION_DEPTH * MAX_CONVERSATION_STREAMS


@pytest.mark.asyncio
async def test_get_target_response_and_store(orchestrator: PAIROrchestrator) -> None:
    # Setup
    sample_text = "Test prompt"
    expected_response = PromptRequestResponse(
        request_pieces=[PromptRequestPiece(original_value=sample_text, converted_value=sample_text, role="user")]
    )
    expected_conversation_id = "12345678-1234-5678-1234-567812345678"

    with patch("uuid.uuid4", return_value=uuid.UUID(expected_conversation_id)):
        orchestrator._prompt_normalizer.send_prompt_async = AsyncMock(return_value=expected_response)  # type: ignore

        response = await orchestrator._get_target_response_and_store(text=sample_text)

        assert response == expected_response

        orchestrator._prompt_normalizer.send_prompt_async.assert_called_once()
        normalizer_request = orchestrator._prompt_normalizer.send_prompt_async.call_args[1]["normalizer_request"]

        assert normalizer_request.conversation_id == expected_conversation_id

        orchestrator._prompt_normalizer.send_prompt_async.assert_called_with(
            normalizer_request=ANY,  # We already checked the conversation_id separately, so use ANY here
            target=orchestrator._prompt_target,
            labels=orchestrator._global_memory_labels,
            orchestrator_identifier=orchestrator.get_identifier(),
        )


@pytest.mark.asyncio
async def test_start_new_conversation(
    orchestrator: PAIROrchestrator, correctly_formatted_response_piece: PromptRequestPiece
) -> None:
    with patch.object(orchestrator, "_prompt_normalizer") as mock_normalizer:
        mock_normalizer.send_prompt_async = AsyncMock(
            return_value=PromptRequestResponse(request_pieces=[correctly_formatted_response_piece])
        )
        await orchestrator._get_attacker_response_and_store(target_response="response", start_new_conversation=True)
        assert orchestrator._last_attacker_conversation_id != ""
        mock_normalizer.send_prompt_async.assert_called_once()


@pytest.mark.asyncio
async def test_continue_conversation(
    orchestrator: PAIROrchestrator, correctly_formatted_response_piece: PromptRequestPiece
) -> None:
    orchestrator._last_attacker_conversation_id = "existing_id"
    with patch.object(orchestrator, "_prompt_normalizer") as mock_normalizer:
        mock_normalizer.send_prompt_async = AsyncMock(
            return_value=PromptRequestResponse(request_pieces=[correctly_formatted_response_piece])
        )
        await orchestrator._get_attacker_response_and_store(target_response="response", start_new_conversation=False)
        assert orchestrator._last_attacker_conversation_id == "existing_id"
        mock_normalizer.send_prompt_async.assert_called_once()


@pytest.mark.asyncio
async def test_attacker_response(
    orchestrator: PAIROrchestrator, correctly_formatted_response_piece: PromptRequestPiece
) -> None:
    expected_response = PromptRequestResponse(request_pieces=[correctly_formatted_response_piece])
    with patch.object(orchestrator, "_prompt_normalizer") as mock_normalizer:
        mock_normalizer.send_prompt_async = AsyncMock(return_value=expected_response)
        response = await orchestrator._get_attacker_response_and_store(
            target_response="response", start_new_conversation=False
        )
        assert response == "prompt"
