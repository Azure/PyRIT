# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import uuid
from typing import Generator
from unittest.mock import Mock, AsyncMock, ANY
from unittest.mock import patch

import pytest

from pyrit.memory import MemoryInterface
from pyrit.models import PromptRequestResponse, Score
from pyrit.orchestrator import PairOrchestrator
from pyrit.orchestrator.pair_orchestrator import PromptRequestPiece
from pyrit.prompt_target import AzureOpenAIChatTarget
from pyrit.score import Scorer
from tests.mocks import get_memory_interface


def _build_prompt_response_with_single_prompt_piece(*, prompt: str) -> PromptRequestResponse:
    return PromptRequestResponse(
        request_pieces=[PromptRequestPiece(original_value=prompt, converted_value=prompt, role="user")]
    )


@pytest.fixture
def memory_interface() -> Generator[MemoryInterface, None, None]:
    yield from get_memory_interface()


@pytest.fixture
def chat_completion_engine() -> AzureOpenAIChatTarget:
    return AzureOpenAIChatTarget(deployment_name="test", endpoint="test", api_key="test")


@pytest.fixture
def scorer_mock(memory_interface: MemoryInterface) -> Scorer:
    scorer = Mock()
    scorer.scorer_type = "float_scale"
    scorer.score_async = AsyncMock(return_value=[])
    return scorer


@pytest.fixture
def orchestrator(memory_interface: MemoryInterface, scorer_mock: Scorer) -> PairOrchestrator:
    target = Mock()
    attacker = Mock()
    orchestrator = PairOrchestrator(
        prompt_target=target,
        desired_target_response_prefix="desired response",
        red_teaming_chat=attacker,
        conversation_objective="attacker objective",
        memory=memory_interface,
        scorer=scorer_mock,
    )

    return orchestrator


@pytest.mark.asyncio
async def test_init(orchestrator):
    assert orchestrator._target_text_model is not None
    assert orchestrator._attacker_text_model is not None
    assert orchestrator._scorer is not None
    assert orchestrator._attacker_objective == "attacker objective"
    assert orchestrator._desired_target_response_prefix == "desired response"


@pytest.mark.asyncio
async def test_run(orchestrator: PairOrchestrator):
    orchestrator._process_conversation_stream = AsyncMock(return_value=[])  # type: ignore
    orchestrator._should_stop = Mock(return_value=False)  # type: ignore
    result = await orchestrator.run()
    assert result == []
    orchestrator._process_conversation_stream.assert_called()
    orchestrator._should_stop.assert_called()


@pytest.mark.asyncio
async def test_output_is_properly_formatted_when_jailbreak_is_found(
    orchestrator: PairOrchestrator,
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
async def test_output_is_properly_formatted_when_jailbreak_is_not_found(
    orchestrator: PairOrchestrator,
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
async def test_get_target_response_and_store(orchestrator: PairOrchestrator) -> None:
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
        orchestrator._prompt_normalizer.send_prompt_async.assert_called_with(
            normalizer_request=ANY,  # Use ANY from unittest.mock to ignore specific instance comparison
            target=orchestrator._target_text_model,
            conversation_id=expected_conversation_id,
            labels=orchestrator._global_memory_labels,
            orchestrator_identifier=orchestrator.get_identifier(),
        )


@pytest.mark.asyncio
async def test_start_new_conversation(orchestrator: PairOrchestrator) -> None:
    with patch.object(orchestrator, "_prompt_normalizer") as mock_normalizer:
        mock_normalizer.send_prompt_async = AsyncMock(return_value=PromptRequestResponse(request_pieces=[]))
        await orchestrator._get_attacker_response_and_store(target_response="response", start_new_conversation=True)
        assert orchestrator._last_attacker_conversation_id != ""
        mock_normalizer.send_prompt_async.assert_called_once()


@pytest.mark.asyncio
async def test_continue_conversation(orchestrator: PairOrchestrator) -> None:
    orchestrator._last_attacker_conversation_id = "existing_id"
    with patch.object(orchestrator, "_prompt_normalizer") as mock_normalizer:
        mock_normalizer.send_prompt_async = AsyncMock(return_value=PromptRequestResponse(request_pieces=[]))
        await orchestrator._get_attacker_response_and_store(target_response="response", start_new_conversation=False)
        assert orchestrator._last_attacker_conversation_id == "existing_id"
        mock_normalizer.send_prompt_async.assert_called_once()


@pytest.mark.asyncio
async def test_attacker_response(orchestrator: PairOrchestrator) -> None:
    expected_response = PromptRequestResponse(request_pieces=[])
    with patch.object(orchestrator, "_prompt_normalizer") as mock_normalizer:
        mock_normalizer.send_prompt_async = AsyncMock(return_value=expected_response)
        response = await orchestrator._get_attacker_response_and_store(
            target_response="response", start_new_conversation=False
        )
        assert response == expected_response
