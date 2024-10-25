# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import pathlib
from pyrit.models import PromptRequestResponse, PromptRequestPiece
import pytest

from typing import Dict, Generator, List, Union
from unittest.mock import AsyncMock, MagicMock, Mock, patch
from uuid import uuid4

from pyrit.memory import MemoryInterface
from pyrit.prompt_target import PromptTarget, OpenAIChatTarget
from pyrit.orchestrator import RedTeamingOrchestrator
from pyrit.models import AttackStrategy
from pyrit.score import Score, Scorer
from pyrit.common.path import DATASETS_PATH

from tests.mocks import get_memory_interface


@pytest.fixture
def memory_interface() -> Generator[MemoryInterface, None, None]:
    yield from get_memory_interface()


@pytest.fixture
def chat_completion_engine() -> OpenAIChatTarget:
    return OpenAIChatTarget(deployment_name="test", endpoint="test", api_key="test")


@pytest.fixture
def prompt_target(memory_interface) -> OpenAIChatTarget:
    return OpenAIChatTarget(
        deployment_name="test",
        endpoint="test",
        api_key="test",
        memory=memory_interface,
    )


@pytest.fixture
def simple_attack_strategy() -> AttackStrategy:
    return AttackStrategy(
        strategy=pathlib.Path(DATASETS_PATH) / "orchestrators" / "red_teaming" / "text_generation.yaml",
        conversation_objective="Do bad stuff",
    )


def _check_orchestrator_memory(memory, num_turns: int):
    conversations = memory.get_all_prompt_pieces()
    # one turn has system prompt, req/resp to target, req/resp to red team target
    expected_num_memories = (4 * num_turns) + 1

    assert len(conversations) == expected_num_memories
    _check_two_conversation_ids(conversations)


def _check_two_conversation_ids(conversations):
    grouped_conversations: Dict[str, List[str]] = {}  # type: ignore
    for obj in conversations:
        key = obj.conversation_id
        if key in grouped_conversations:
            grouped_conversations[key].append(obj)
        else:
            grouped_conversations[key] = [obj]

    assert (
        len(grouped_conversations.keys()) == 2
    ), "There should be two conversation threads, one with target and one with rt target"


@pytest.mark.asyncio
@pytest.mark.parametrize("attack_strategy_as_str", [True, False])
async def test_send_prompt_twice(
    prompt_target: PromptTarget,
    chat_completion_engine: OpenAIChatTarget,
    simple_attack_strategy: AttackStrategy,
    memory_interface: MemoryInterface,
    attack_strategy_as_str: bool,
):
    attack_strategy: Union[str | AttackStrategy] = (
        str(simple_attack_strategy) if attack_strategy_as_str else simple_attack_strategy
    )

    scorer = MagicMock(Scorer)
    scorer.scorer_type = "true_false"
    red_teaming_orchestrator = RedTeamingOrchestrator(
        red_teaming_chat=chat_completion_engine,
        memory=memory_interface,
        attack_strategy=attack_strategy,
        prompt_target=prompt_target,
        objective_scorer=scorer,
    )

    prompt_target_conversation_id = str(uuid4())
    red_teaming_chat_conversation_id = str(uuid4())
    with patch.object(red_teaming_orchestrator._red_teaming_chat, "_complete_chat_async") as mock_rt:
        with patch.object(red_teaming_orchestrator._prompt_target, "_complete_chat_async") as mock_target:
            mock_rt.return_value = "First red teaming chat response"
            expected_target_response = "First target response"
            mock_target.return_value = expected_target_response
            target_response = await red_teaming_orchestrator._retrieve_and_send_prompt_async(
                prompt_target_conversation_id=prompt_target_conversation_id,
                red_teaming_chat_conversation_id=red_teaming_chat_conversation_id,
            )
            assert target_response.converted_value == expected_target_response

            _check_orchestrator_memory(memory=red_teaming_orchestrator._memory, num_turns=1)

            mock_rt.assert_called_once()
            mock_target.assert_called_once()

            second_target_response = "Second target response"
            mock_rt.return_value = "Second red teaming chat response"
            mock_target.return_value = second_target_response
            target_response = await red_teaming_orchestrator._retrieve_and_send_prompt_async(
                prompt_target_conversation_id=prompt_target_conversation_id,
                red_teaming_chat_conversation_id=red_teaming_chat_conversation_id,
            )
            assert target_response.converted_value == second_target_response

            _check_orchestrator_memory(memory=red_teaming_orchestrator._memory, num_turns=2)


@pytest.mark.asyncio
@pytest.mark.parametrize("score", [True, False])
@pytest.mark.parametrize("message_count", [0, 2, 4, 6, 10, 20, 100])
async def test_is_conversation_complete_scoring(score, message_count):
    mock_score = MagicMock(Score)
    mock_score.score_value = score
    mock_score.score_type = "true_false"
    mock_score.get_value.return_value = score

    mock_scorer = MagicMock(Scorer)
    mock_scorer.scorer_type = "true_false"
    mock_scorer.score_async = AsyncMock(return_value=[mock_score])

    orchestrator = RedTeamingOrchestrator(
        attack_strategy="some strategy",
        red_teaming_chat=Mock(),
        prompt_target=Mock(),
        memory=Mock(),
        objective_scorer=mock_scorer,
    )
    # simulate back and forth between user and assistant
    simulated_messages = [
        PromptRequestResponse(
            request_pieces=[
                PromptRequestPiece(
                    role="user" if i % 2 == 0 else "assistant",
                    original_value=f"Message #{i}",
                    converted_value=f"Message #{i}",
                    original_value_data_type="text",
                    converted_value_data_type="text",
                )
            ]
        )
        for i in range(message_count)
    ]
    orchestrator._memory.get_conversation = MagicMock(return_value=simulated_messages)
    # conversation is complete if the last message is from the target
    # and the score is True
    actual_result = await orchestrator._check_conversation_complete_async(prompt_target_conversation_id=str(uuid4()))
    is_failure = not bool(actual_result) or not actual_result.score_value
    assert not is_failure == (len(simulated_messages) > 0 and score)


@pytest.mark.asyncio
async def test_is_conversation_complete_scoring_non_bool():
    mock_score = MagicMock(Score)
    mock_score.score_type = "float_scale"
    mock_score.score_value = 0.5

    scorer = MagicMock(Scorer)
    scorer.scorer_type = "true_false"
    scorer.score_text_async = AsyncMock(return_value=[mock_score])

    orchestrator = RedTeamingOrchestrator(
        attack_strategy="some strategy",
        red_teaming_chat=Mock(),
        prompt_target=Mock(),
        memory=Mock(),
        objective_scorer=scorer,
    )
    orchestrator._memory.get_conversation = MagicMock(
        return_value=[
            PromptRequestResponse(
                request_pieces=[
                    PromptRequestPiece(
                        role="user",
                        original_value="First message.",
                        converted_value="First message.",
                        original_value_data_type="text",
                        converted_value_data_type="text",
                    )
                ]
            ),
            PromptRequestResponse(
                request_pieces=[
                    PromptRequestPiece(
                        role="assistant",
                        original_value="Second message.",
                        converted_value="Second message.",
                        original_value_data_type="text",
                        converted_value_data_type="text",
                    )
                ]
            ),
            PromptRequestResponse(
                request_pieces=[
                    PromptRequestPiece(
                        role="user",
                        original_value="Third message.",
                        converted_value="Third message.",
                        original_value_data_type="text",
                        converted_value_data_type="text",
                    )
                ]
            ),
            PromptRequestResponse(
                request_pieces=[
                    PromptRequestPiece(
                        role="assistant",
                        original_value="Fourth message.",
                        converted_value="Fourth message.",
                        original_value_data_type="text",
                        converted_value_data_type="text",
                    )
                ]
            ),
        ]
    )
    with pytest.raises(ValueError):
        await orchestrator._check_conversation_complete_async(prompt_target_conversation_id=str(uuid4()))


@pytest.mark.asyncio
@pytest.mark.parametrize("max_turns", [1, 3, 5])
async def test_apply_attack_strategy_until_completion_async(
    prompt_target: PromptTarget,
    chat_completion_engine: OpenAIChatTarget,
    simple_attack_strategy: AttackStrategy,
    memory_interface: MemoryInterface,
    max_turns: int,
):
    scorer = MagicMock(Scorer)
    scorer.scorer_type = "true_false"
    red_teaming_orchestrator = RedTeamingOrchestrator(
        red_teaming_chat=chat_completion_engine,
        memory=memory_interface,
        attack_strategy=simple_attack_strategy,
        prompt_target=prompt_target,
        objective_scorer=scorer,
    )

    with (
        patch.object(red_teaming_orchestrator, "_retrieve_and_send_prompt_async") as mock_send_prompt,
        patch.object(red_teaming_orchestrator, "_check_conversation_complete_async") as mock_check_complete,
    ):

        mock_send_prompt.return_value = MagicMock(response_error="none")
        mock_check_complete.return_value = MagicMock(get_value=MagicMock(return_value=True))

        conversation_id = await red_teaming_orchestrator.apply_attack_async(
            max_turns=max_turns
        )

        assert conversation_id is not None
        assert red_teaming_orchestrator._achieved_objective is True
        assert mock_send_prompt.call_count <= max_turns
        assert mock_check_complete.call_count <= max_turns


@pytest.mark.asyncio
async def test_apply_attack_strategy_until_completion_async_blocked_response(
    prompt_target: PromptTarget,
    chat_completion_engine: OpenAIChatTarget,
    simple_attack_strategy: AttackStrategy,
    memory_interface: MemoryInterface,
):
    scorer = MagicMock(Scorer)
    scorer.scorer_type = "true_false"
    red_teaming_orchestrator = RedTeamingOrchestrator(
        red_teaming_chat=chat_completion_engine,
        memory=memory_interface,
        attack_strategy=simple_attack_strategy,
        prompt_target=prompt_target,
        objective_scorer=scorer,
    )

    with patch.object(red_teaming_orchestrator, "_retrieve_and_send_prompt_async") as mock_send_prompt:
        mock_send_prompt.return_value = MagicMock(response_error="blocked")

        conversation_id = await red_teaming_orchestrator.apply_attack_async(max_turns=5)

        assert conversation_id is not None
        assert red_teaming_orchestrator._achieved_objective is False
        assert mock_send_prompt.call_count == 5


@pytest.mark.asyncio
async def test_apply_attack_strategy_until_completion_async_runtime_error(
    prompt_target: PromptTarget,
    chat_completion_engine: OpenAIChatTarget,
    simple_attack_strategy: AttackStrategy,
    memory_interface: MemoryInterface,
):
    scorer = MagicMock(Scorer)
    scorer.scorer_type = "true_false"
    red_teaming_orchestrator = RedTeamingOrchestrator(
        red_teaming_chat=chat_completion_engine,
        memory=memory_interface,
        attack_strategy=simple_attack_strategy,
        prompt_target=prompt_target,
        objective_scorer=scorer,
    )

    with patch.object(red_teaming_orchestrator, "_retrieve_and_send_prompt_async") as mock_send_prompt:
        mock_send_prompt.return_value = MagicMock(response_error="unexpected_error")

        with pytest.raises(RuntimeError):
            await red_teaming_orchestrator.apply_attack_async(max_turns=5)
