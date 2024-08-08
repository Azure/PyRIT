# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import pathlib
from pyrit.models import PromptRequestResponse, PromptRequestPiece
import pytest

from typing import Dict, Generator, List, Union
from unittest.mock import AsyncMock, MagicMock, Mock, patch

from pyrit.memory import MemoryInterface
from pyrit.prompt_target import PromptTarget, AzureOpenAITextChatTarget
from pyrit.orchestrator import RedTeamingOrchestrator
from pyrit.models import AttackStrategy
from pyrit.score import Score, Scorer
from pyrit.common.path import DATASETS_PATH

from tests.mocks import get_memory_interface


@pytest.fixture
def memory_interface() -> Generator[MemoryInterface, None, None]:
    yield from get_memory_interface()


@pytest.fixture
def chat_completion_engine() -> AzureOpenAITextChatTarget:
    return AzureOpenAITextChatTarget(deployment_name="test", endpoint="test", api_key="test")


@pytest.fixture
def prompt_target(memory_interface) -> AzureOpenAITextChatTarget:
    return AzureOpenAITextChatTarget(
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


def _check_orchestrator_memory_if_no_original_prompt(memory, num_turns: int):
    conversations = memory.get_all_prompt_pieces()
    # one turn has system prompt, req/resp to target, req/resp to red team target
    expected_num_memories = (4 * num_turns) + 1

    assert len(conversations) == expected_num_memories
    _check_two_conversation_ids(conversations)


def _check_orchestrator_memory_if_original_prompt(memory, num_turns: int):
    conversations = memory.get_all_prompt_pieces()

    if num_turns == 1:
        assert len(conversations) == 2, "prompt was supplied, there should be 1 req/response"
        assert conversations[0].conversation_id == conversations[1].conversation_id
    else:
        # one turn has system prompt, req/resp to target, req/resp to red team target
        # except the first turn is just between a supplied prompt and a target
        expected_num_memories = (4 * num_turns) - 1

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
    chat_completion_engine: AzureOpenAITextChatTarget,
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
        initial_red_teaming_prompt="how can I help you?",
        prompt_target=prompt_target,
        scorer=scorer,
    )

    with patch.object(red_teaming_orchestrator._red_teaming_chat, "_complete_chat_async") as mock_rt:
        with patch.object(red_teaming_orchestrator._prompt_target, "_complete_chat_async") as mock_target:
            mock_rt.return_value = "First red teaming chat response"
            expected_target_response = "First target response"
            mock_target.return_value = expected_target_response
            target_response = await red_teaming_orchestrator.send_prompt_async()
            assert target_response.converted_value == expected_target_response

            _check_orchestrator_memory_if_no_original_prompt(memory=red_teaming_orchestrator._memory, num_turns=1)

            mock_rt.assert_called_once()
            mock_target.assert_called_once()

            second_target_response = "Second target response"
            mock_rt.return_value = "Second red teaming chat response"
            mock_target.return_value = second_target_response
            target_response = await red_teaming_orchestrator.send_prompt_async()
            assert target_response.converted_value == second_target_response

            _check_orchestrator_memory_if_no_original_prompt(memory=red_teaming_orchestrator._memory, num_turns=2)


@pytest.mark.asyncio
@pytest.mark.parametrize("attack_strategy_as_str", [True, False])
async def test_send_fixed_prompt_then_generated_prompt(
    prompt_target: PromptTarget,
    chat_completion_engine: AzureOpenAITextChatTarget,
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
        initial_red_teaming_prompt="how can I help you?",
        prompt_target=prompt_target,
        scorer=scorer,
    )

    with patch.object(red_teaming_orchestrator._red_teaming_chat, "_complete_chat_async") as mock_rt:
        with patch.object(red_teaming_orchestrator._prompt_target, "_complete_chat_async") as mock_target:
            fixed_input_prompt = "First prompt to target - set by user"
            expected_target_responses = "First target response"
            mock_target.return_value = expected_target_responses
            target_response = await red_teaming_orchestrator.send_prompt_async(prompt=fixed_input_prompt)
            assert target_response.converted_value == expected_target_responses

            _check_orchestrator_memory_if_original_prompt(memory=red_teaming_orchestrator._memory, num_turns=1)

            mock_rt.assert_not_called()
            mock_target.assert_called_once()

            expected_generated_red_teaming_response = "red teaming chat response"
            expected_target_response = "second chat response"

            mock_rt.return_value = expected_generated_red_teaming_response
            mock_target.return_value = expected_target_response

            target_response = await red_teaming_orchestrator.send_prompt_async()
            assert target_response.converted_value == expected_target_response

            _check_orchestrator_memory_if_original_prompt(memory=red_teaming_orchestrator._memory, num_turns=2)


@pytest.mark.asyncio
@pytest.mark.parametrize("attack_strategy_as_str", [True, False])
async def test_send_fixed_prompt_memory_labels(
    prompt_target: PromptTarget,
    chat_completion_engine: AzureOpenAITextChatTarget,
    simple_attack_strategy: AttackStrategy,
    memory_interface: MemoryInterface,
    attack_strategy_as_str: bool,
):
    attack_strategy: Union[str | AttackStrategy] = (
        str(simple_attack_strategy) if attack_strategy_as_str else simple_attack_strategy
    )

    scorer = MagicMock(Scorer)
    scorer.scorer_type = "true_false"
    labels = {"op_name": "op1", "user_name": "name1"}
    red_teaming_orchestrator = RedTeamingOrchestrator(
        red_teaming_chat=chat_completion_engine,
        memory=memory_interface,
        memory_labels=labels,
        attack_strategy=attack_strategy,
        initial_red_teaming_prompt="how can I help you?",
        prompt_target=prompt_target,
        scorer=scorer,
    )

    with patch.object(red_teaming_orchestrator._red_teaming_chat, "_complete_chat_async") as _:
        with patch.object(red_teaming_orchestrator._prompt_target, "_complete_chat_async") as mock_target:
            fixed_input_prompt = "First prompt to target - set by user"
            expected_target_responses = "First target response"
            mock_target.return_value = expected_target_responses
            target_response = await red_teaming_orchestrator.send_prompt_async(prompt=fixed_input_prompt)
            assert target_response.labels == labels


@pytest.mark.asyncio
@pytest.mark.parametrize("attack_strategy_as_str", [True, False])
async def test_send_fixed_prompt_beyond_first_iteration_failure(
    prompt_target: PromptTarget,
    chat_completion_engine: AzureOpenAITextChatTarget,
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
        initial_red_teaming_prompt="how can I help you?",
        prompt_target=prompt_target,
        scorer=scorer,
    )

    with patch.object(red_teaming_orchestrator._red_teaming_chat, "_complete_chat_async") as mock_rt:
        with patch.object(red_teaming_orchestrator._prompt_target, "_complete_chat_async") as mock_target:
            fixed_input_prompt = "First prompt to target - set by user"
            expected_target_response = "First target response"
            mock_target.return_value = expected_target_response
            target_response = await red_teaming_orchestrator.send_prompt_async(prompt=fixed_input_prompt)
            assert target_response.converted_value == expected_target_response
            _check_orchestrator_memory_if_original_prompt(memory=red_teaming_orchestrator._memory, num_turns=1)

            mock_rt.assert_not_called()
            mock_target.assert_called_once()

            second_fixed_input_prompt = "Second prompt to target - sent by user"
            expected_target_response = "Second target response"
            mock_target.return_value = expected_target_response
            with pytest.raises(ValueError):
                target_response = await red_teaming_orchestrator.send_prompt_async(prompt=second_fixed_input_prompt)
            mock_rt.assert_not_called()


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
        scorer=mock_scorer,
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
    actual_result = await orchestrator.check_conversation_complete_async()
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
        scorer=scorer,
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
        await orchestrator.check_conversation_complete_async()
