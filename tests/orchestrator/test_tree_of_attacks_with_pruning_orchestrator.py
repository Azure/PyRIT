# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import pytest
import tempfile

from typing import Generator, Union
from unittest.mock import AsyncMock, MagicMock, Mock, patch

from pyrit.memory import MemoryInterface
from pyrit.models import PromptRequestResponse, PromptRequestPiece
from pyrit.prompt_target import PromptTarget
from pyrit.orchestrator import TreeOfAttacksWithPruningOrchestrator
from pyrit.models import AttackStrategy
from pyrit.score import Score, Scorer
from tests.mocks import MockPromptTarget, get_memory_interface


@pytest.fixture
def memory() -> Generator[MemoryInterface, None, None]:
    yield from get_memory_interface()


def create_target():
    return MockPromptTarget(memory=memory)

@pytest.fixture
def prompt_target(memory) -> MockPromptTarget:
    fd, path = tempfile.mkstemp(suffix=".json.memory")
    return MockPromptTarget(memory=memory)

@pytest.fixture
def red_teaming_target(memory) -> MockPromptTarget:
    fd, path = tempfile.mkstemp(suffix=".json.memory")
    return MockPromptTarget(memory=memory)

@pytest.fixture
def scoring_target(memory) -> MockPromptTarget:
    fd, path = tempfile.mkstemp(suffix=".json.memory")
    return MockPromptTarget(memory=memory)


# various value for width, branching factor, depth
# tests for exception types
# with and without on-topic checking enabled (count calls to scorer target)
# with and without converters

def get_prompt_response_with_content(content: str) -> PromptRequestResponse:
    return PromptRequestResponse(
        request_pieces=[
            PromptRequestPiece(
                role="assistant",
                original_value=content,
                converted_value=content,
                original_value_data_type="text",
                converted_value_data_type="text",
            )
        ]
    )

def test_invalid_width():
    with pytest.raises(ValueError) as e:
        TreeOfAttacksWithPruningOrchestrator(
            red_teaming_chat=MagicMock(),
            memory=MagicMock(),
            prompt_target=MagicMock(),
            scoring_target=MagicMock(),
            width=0,
            branching_factor=2,
            depth=2,
            conversation_objective="conversation objective",
            on_topic_checking_enabled=True,
        )
    assert e.match("The width of the tree must be at least 1.")


def test_invalid_branching_factor():
    with pytest.raises(ValueError) as e:
        TreeOfAttacksWithPruningOrchestrator(
            red_teaming_chat=MagicMock(),
            memory=MagicMock(),
            prompt_target=MagicMock(),
            scoring_target=MagicMock(),
            width=4,
            branching_factor=0,
            depth=2,
            conversation_objective="conversation objective",
            on_topic_checking_enabled=True,
        )
    assert e.match("The branching factor of the tree must be at least 1.")


def test_invalid_depth():
    with pytest.raises(ValueError) as e:
        TreeOfAttacksWithPruningOrchestrator(
            red_teaming_chat=MagicMock(),
            memory=MagicMock(),
            prompt_target=MagicMock(),
            scoring_target=MagicMock(),
            width=4,
            branching_factor=3,
            depth=0,
            conversation_objective="conversation objective",
            on_topic_checking_enabled=True,
        )
    assert e.match("The depth of the tree must be at least 1.")


def _check_orchestrator_memory(memory, num_turns: int):
    conversations = memory.get_all_prompt_pieces()
    # one turn has system prompt for red team target, req/resp to red team target,
    # req/resp to target,
    # system prompt for scoring target,
    # system prompt for on-topic checking target
    expected_num_memories = (4 * num_turns) + 3

    assert len(conversations) == expected_num_memories


@pytest.mark.asyncio
async def test_apply_strategy_single_turn_success(
    prompt_target: MockPromptTarget,
    red_teaming_target: MockPromptTarget,
    memory: MemoryInterface,
    scoring_target: MockPromptTarget,
):
    tap_orchestrator = TreeOfAttacksWithPruningOrchestrator(
        red_teaming_chat=red_teaming_target,
        memory=memory,
        prompt_target=prompt_target,
        scoring_target=scoring_target,
        width=1,
        branching_factor=1,
        depth=1,
        conversation_objective="conversation objective",
        on_topic_checking_enabled=True,
        verbose=True
    )

    with patch.object(tap_orchestrator._red_teaming_chat, "send_prompt_async") as mock_red_teaming_target:
        with patch.object(tap_orchestrator._scoring_target, "send_prompt_async") as mock_scoring_target:
            with patch.object(tap_orchestrator._prompt_target, "send_prompt_async") as mock_target:
                mock_red_teaming_target.return_value = get_prompt_response_with_content(
                    content='{"improvements": "First improvement response", "prompt": "First red teaming prompt"}',
                )
                mock_scoring_target.side_effect = [
                    get_prompt_response_with_content(
                        '{"value": "True", "description": "On topic", "rationale": "Rationale"}'),
                    get_prompt_response_with_content(
                        '{"score_value": "5", "description": "Jailbreak", "rationale": "Rationale"}'),
                ]
                expected_target_response = get_prompt_response_with_content("First target response")
                mock_target.return_value = expected_target_response

                result = await tap_orchestrator.apply_attack_strategy_async()
                
                assert len(tap_orchestrator._orchestrators) == 1
                
                assert result is not None
                assert not result.pruned
                assert result.completed
                assert result.score == 1  # (5-1) / 4
                assert result.orchestrator_id == tap_orchestrator._orchestrators[0].get_identifier()["id"]
                assert result.prompt_target_conversation_id is not None

                mock_red_teaming_target.assert_called_once()
                mock_target.assert_called_once()
                assert mock_scoring_target.call_count == 2

                _check_orchestrator_memory(memory=tap_orchestrator._memory, num_turns=1)


@pytest.mark.asyncio
async def test_apply_strategy_max_depth_reached(
    prompt_target: MockPromptTarget,
    red_teaming_target: MockPromptTarget,
    memory: MemoryInterface,
    scoring_target: MockPromptTarget,
):
    tap_orchestrator = TreeOfAttacksWithPruningOrchestrator(
        red_teaming_chat=red_teaming_target,
        memory=memory,
        prompt_target=prompt_target,
        scoring_target=scoring_target,
        width=1,
        branching_factor=1,
        depth=1,
        conversation_objective="conversation objective",
        on_topic_checking_enabled=True,
        verbose=True
    )

    with patch.object(tap_orchestrator._red_teaming_chat, "send_prompt_async") as mock_red_teaming_target:
        with patch.object(tap_orchestrator._scoring_target, "send_prompt_async") as mock_scoring_target:
            with patch.object(tap_orchestrator._prompt_target, "send_prompt_async") as mock_target:
                mock_red_teaming_target.return_value = get_prompt_response_with_content(
                    content='{"improvements": "First improvement response", "prompt": "First red teaming prompt"}',
                )
                mock_scoring_target.side_effect = [
                    get_prompt_response_with_content(
                        '{"value": "True", "description": "On topic", "rationale": "Rationale"}'),
                    get_prompt_response_with_content(
                        '{"score_value": "3", "description": "No jailbreak", "rationale": "Rationale"}'),
                ]
                expected_target_response = get_prompt_response_with_content("First target response")
                mock_target.return_value = expected_target_response

                result = await tap_orchestrator.apply_attack_strategy_async()
                
                assert len(tap_orchestrator._orchestrators) == 1
                
                assert result is None

                mock_red_teaming_target.assert_called_once()
                mock_target.assert_called_once()
                assert mock_scoring_target.call_count == 2

                _check_orchestrator_memory(memory=tap_orchestrator._memory, num_turns=1)


@pytest.mark.asyncio
@pytest.mark.parametrize("attack_strategy_as_str", [True, False])
async def test_send_fixed_prompt_then_generated_prompt(
    prompt_target: PromptTarget,
    simple_attack_strategy: AttackStrategy,
    memory: MemoryInterface,
    attack_strategy_as_str: bool,
):
    attack_strategy: Union[str | AttackStrategy] = (
        str(simple_attack_strategy) if attack_strategy_as_str else simple_attack_strategy
    )

    scorer = MagicMock(Scorer)
    scorer.scorer_type = "true_false"
    red_teaming_orchestrator = RedTeamingOrchestrator(
        red_teaming_chat=chat_completion_engine,
        memory=memory,
        attack_strategy=attack_strategy,
        initial_red_teaming_prompt="how can I help you?",
        prompt_target=prompt_target,
        scorer=scorer,
    )

    with patch.object(red_teaming_orchestrator._red_teaming_chat, "_complete_chat_async") as mock_red_teaming_target:
        with patch.object(red_teaming_orchestrator._prompt_target, "_complete_chat_async") as mock_target:
            fixed_input_prompt = "First prompt to target - set by user"
            expected_target_responses = "First target response"
            mock_target.return_value = expected_target_responses
            target_response = await red_teaming_orchestrator.send_prompt_async(prompt=fixed_input_prompt)
            assert target_response.converted_value == expected_target_responses

            _check_orchestrator_memory_if_original_prompt(memory=red_teaming_orchestrator._memory, num_turns=1)

            mock_red_teaming_target.assert_not_called()
            mock_target.assert_called_once()

            expected_generated_red_teaming_response = "red teaming chat response"
            expected_target_response = "second chat response"

            mock_red_teaming_target.return_value = expected_generated_red_teaming_response
            mock_target.return_value = expected_target_response

            target_response = await red_teaming_orchestrator.send_prompt_async()
            assert target_response.converted_value == expected_target_response

            _check_orchestrator_memory_if_original_prompt(memory=red_teaming_orchestrator._memory, num_turns=2)


@pytest.mark.asyncio
@pytest.mark.parametrize("attack_strategy_as_str", [True, False])
async def test_send_fixed_prompt_beyond_first_iteration_failure(
    prompt_target: PromptTarget,
    simple_attack_strategy: AttackStrategy,
    memory: MemoryInterface,
    attack_strategy_as_str: bool,
):
    attack_strategy: Union[str | AttackStrategy] = (
        str(simple_attack_strategy) if attack_strategy_as_str else simple_attack_strategy
    )

    scorer = MagicMock(Scorer)
    scorer.scorer_type = "true_false"
    red_teaming_orchestrator = RedTeamingOrchestrator(
        red_teaming_chat=chat_completion_engine,
        memory=memory,
        attack_strategy=attack_strategy,
        initial_red_teaming_prompt="how can I help you?",
        prompt_target=prompt_target,
        scorer=scorer,
    )

    with patch.object(red_teaming_orchestrator._red_teaming_chat, "_complete_chat_async") as mock_red_teaming_target:
        with patch.object(red_teaming_orchestrator._prompt_target, "_complete_chat_async") as mock_target:
            fixed_input_prompt = "First prompt to target - set by user"
            expected_target_response = "First target response"
            mock_target.return_value = expected_target_response
            target_response = await red_teaming_orchestrator.send_prompt_async(prompt=fixed_input_prompt)
            assert target_response.converted_value == expected_target_response
            _check_orchestrator_memory_if_original_prompt(memory=red_teaming_orchestrator._memory, num_turns=1)

            mock_red_teaming_target.assert_not_called()
            mock_target.assert_called_once()

            second_fixed_input_prompt = "Second prompt to target - sent by user"
            expected_target_response = "Second target response"
            mock_target.return_value = expected_target_response
            with pytest.raises(ValueError):
                target_response = await red_teaming_orchestrator.send_prompt_async(prompt=second_fixed_input_prompt)
            mock_red_teaming_target.assert_not_called()


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
