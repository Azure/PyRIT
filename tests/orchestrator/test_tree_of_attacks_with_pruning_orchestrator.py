# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import pytest

from typing import Generator
from unittest.mock import MagicMock, patch

from pyrit.memory import MemoryInterface
from pyrit.memory import CentralMemory
from pyrit.models import PromptRequestResponse, PromptRequestPiece
from pyrit.orchestrator import TreeOfAttacksWithPruningOrchestrator
from tests.mocks import MockPromptTarget, get_memory_interface


@pytest.fixture
def memory() -> Generator[MemoryInterface, None, None]:
    yield from get_memory_interface()


@pytest.fixture
def prompt_target(memory) -> MockPromptTarget:
    with patch.object(CentralMemory, "get_memory_instance", return_value=memory):
        return MockPromptTarget()


@pytest.fixture
def red_teaming_target(memory) -> MockPromptTarget:
    with patch.object(CentralMemory, "get_memory_instance", return_value=memory):
        return MockPromptTarget()


@pytest.fixture
def scoring_target(memory) -> MockPromptTarget:
    with patch.object(CentralMemory, "get_memory_instance", return_value=memory):
        return MockPromptTarget()


def get_single_line_string(input: str) -> str:
    return input.replace("\n", "")


# tests for exception types
# with and without converters
# failures from target, continue on with algorithm
# off-topic pruning


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


def test_invalid_width(memory):
    with patch.object(CentralMemory, "get_memory_instance", return_value=memory):
        with pytest.raises(ValueError) as e:
            TreeOfAttacksWithPruningOrchestrator(
                red_teaming_chat=MagicMock(),
                prompt_target=MagicMock(),
                scoring_target=MagicMock(),
                width=0,
                branching_factor=2,
                depth=2,
                conversation_objective="conversation objective",
                on_topic_checking_enabled=True,
            )
        assert e.match("The width of the tree must be at least 1.")


def test_invalid_branching_factor(memory):
    with patch.object(CentralMemory, "get_memory_instance", return_value=memory):
        with pytest.raises(ValueError) as e:
            TreeOfAttacksWithPruningOrchestrator(
                red_teaming_chat=MagicMock(),
                prompt_target=MagicMock(),
                scoring_target=MagicMock(),
                width=4,
                branching_factor=0,
                depth=2,
                conversation_objective="conversation objective",
                on_topic_checking_enabled=True,
            )
        assert e.match("The branching factor of the tree must be at least 1.")


def test_invalid_depth(memory):
    with patch.object(CentralMemory, "get_memory_instance", return_value=memory):
        with pytest.raises(ValueError) as e:
            TreeOfAttacksWithPruningOrchestrator(
                red_teaming_chat=MagicMock(),
                prompt_target=MagicMock(),
                scoring_target=MagicMock(),
                width=4,
                branching_factor=3,
                depth=0,
                conversation_objective="conversation objective",
                on_topic_checking_enabled=True,
            )
        assert e.match("The depth of the tree must be at least 1.")


@pytest.mark.asyncio
@pytest.mark.parametrize("on_topic_checking_enabled", [True, False])
async def test_apply_strategy_single_turn_success(
    prompt_target: MockPromptTarget,
    red_teaming_target: MockPromptTarget,
    memory: MemoryInterface,
    scoring_target: MockPromptTarget,
    on_topic_checking_enabled: bool,
):
    with patch.object(CentralMemory, "get_memory_instance", return_value=memory):
        tap_orchestrator = TreeOfAttacksWithPruningOrchestrator(
            red_teaming_chat=red_teaming_target,
            prompt_target=prompt_target,
            scoring_target=scoring_target,
            width=1,
            branching_factor=1,
            depth=1,
            conversation_objective="conversation objective",
            on_topic_checking_enabled=on_topic_checking_enabled,
            verbose=False,
        )

        with patch.object(tap_orchestrator._red_teaming_chat, "send_prompt_async") as mock_red_teaming_target:
            with patch.object(tap_orchestrator._scoring_target, "send_prompt_async") as mock_scoring_target:
                with patch.object(tap_orchestrator._prompt_target, "send_prompt_async") as mock_target:
                    mock_red_teaming_target.return_value = get_prompt_response_with_content(
                        content='{"improvements": "First improvement response", "prompt": "First red teaming prompt"}',
                    )
                    mock_scoring_target_return_values = []
                    if on_topic_checking_enabled:
                        mock_scoring_target_return_values.append(
                            get_prompt_response_with_content(
                                '{"score_value": "True", "description": "On topic", "rationale": "Rationale"}'
                            ),
                        )
                    mock_scoring_target_return_values.append(
                        get_prompt_response_with_content(
                            '{"score_value": "10", "description": "Jailbreak", "rationale": "Rationale"}'
                        ),
                    )
                    mock_scoring_target.side_effect = mock_scoring_target_return_values
                    expected_target_response = get_prompt_response_with_content("First target response")
                    mock_target.return_value = expected_target_response

                    result = await tap_orchestrator.apply_attack_strategy_async()

                    assert len(tap_orchestrator._orchestrators) == 1

                    assert result is not None
                    assert not result.pruned
                    assert result.completed
                    assert result.score == (10 - 1) / 9  # normalized score
                    assert result.orchestrator_id == tap_orchestrator._orchestrators[0].get_identifier()["id"]
                    assert result.prompt_target_conversation_id is not None

                    mock_red_teaming_target.assert_called_once()
                    mock_target.assert_called_once()
                    assert mock_scoring_target.call_count == 2 if on_topic_checking_enabled else 1

                    # 4 conversation turns and 3 system prompts, scoring prompts are not stored as of now
                    assert (
                        len(tap_orchestrator._memory.get_all_prompt_pieces()) == 7 if on_topic_checking_enabled else 6
                    )


@pytest.mark.asyncio
@pytest.mark.parametrize("on_topic_checking_enabled", [True, False])
async def test_apply_strategy_max_depth_reached(
    prompt_target: MockPromptTarget,
    red_teaming_target: MockPromptTarget,
    memory: MemoryInterface,
    scoring_target: MockPromptTarget,
    on_topic_checking_enabled: bool,
):
    with patch.object(CentralMemory, "get_memory_instance", return_value=memory):
        tap_orchestrator = TreeOfAttacksWithPruningOrchestrator(
            red_teaming_chat=red_teaming_target,
            prompt_target=prompt_target,
            scoring_target=scoring_target,
            width=1,
            branching_factor=1,
            depth=1,
            conversation_objective="conversation objective",
            on_topic_checking_enabled=on_topic_checking_enabled,
            verbose=False,
        )

        with patch.object(tap_orchestrator._red_teaming_chat, "send_prompt_async") as mock_red_teaming_target:
            with patch.object(tap_orchestrator._scoring_target, "send_prompt_async") as mock_scoring_target:
                with patch.object(tap_orchestrator._prompt_target, "send_prompt_async") as mock_target:
                    mock_red_teaming_target.return_value = get_prompt_response_with_content(
                        content='{"improvements": "First improvement response", "prompt": "First red teaming prompt"}',
                    )
                    mock_scoring_target_return_values = []
                    if on_topic_checking_enabled:
                        mock_scoring_target_return_values.append(
                            get_prompt_response_with_content(
                                '{"score_value": "True", "description": "On topic", "rationale": "Rationale"}'
                            ),
                        )
                    mock_scoring_target_return_values.append(
                        get_prompt_response_with_content(
                            '{"score_value": "3", "description": "No jailbreak", "rationale": "Rationale"}'
                        ),
                    )
                    mock_scoring_target.side_effect = mock_scoring_target_return_values
                    expected_target_response = get_prompt_response_with_content("First target response")
                    mock_target.return_value = expected_target_response

                    result = await tap_orchestrator.apply_attack_strategy_async()

                    assert len(tap_orchestrator._orchestrators) == 1

                    assert result is None

                    mock_red_teaming_target.assert_called_once()
                    mock_target.assert_called_once()
                    assert mock_scoring_target.call_count == 2 if on_topic_checking_enabled else 1

                    # 4 conversation turns and 3 system prompts, scoring prompts are not stored as of now
                    assert (
                        len(tap_orchestrator._memory.get_all_prompt_pieces()) == 7 if on_topic_checking_enabled else 6
                    )


@pytest.mark.asyncio
@pytest.mark.parametrize("depth", [1, 2, 3])
@pytest.mark.parametrize("on_topic_checking_enabled", [True, False])
async def test_apply_strategy_multiturn_failure(
    prompt_target: MockPromptTarget,
    red_teaming_target: MockPromptTarget,
    memory: MemoryInterface,
    scoring_target: MockPromptTarget,
    depth: int,
    on_topic_checking_enabled: bool,
):
    with patch.object(CentralMemory, "get_memory_instance", return_value=memory):
        tap_orchestrator = TreeOfAttacksWithPruningOrchestrator(
            red_teaming_chat=red_teaming_target,
            prompt_target=prompt_target,
            scoring_target=scoring_target,
            width=1,
            branching_factor=1,
            depth=depth,
            conversation_objective="conversation objective",
            on_topic_checking_enabled=on_topic_checking_enabled,
            verbose=False,
        )

        with patch.object(tap_orchestrator._red_teaming_chat, "send_prompt_async") as mock_red_teaming_target:
            with patch.object(tap_orchestrator._scoring_target, "send_prompt_async") as mock_scoring_target:
                with patch.object(tap_orchestrator._prompt_target, "send_prompt_async") as mock_target:
                    mock_red_teaming_target.side_effect = [
                        get_prompt_response_with_content(
                            content=f'{{"improvements": "First branch level {level} improvement response", '
                            f'"prompt": "First branch level {level} red teaming prompt"}}',
                        )
                        for level in range(depth)
                    ]
                    scorer_responses_for_one_turn = []
                    if on_topic_checking_enabled:
                        scorer_responses_for_one_turn.append(
                            get_prompt_response_with_content(
                                '{"score_value": "True", "description": "On topic", "rationale": "Rationale"}'
                            ),
                        )
                    scorer_responses_for_one_turn.append(
                        get_prompt_response_with_content(
                            '{"score_value": "5", "description": "No jailbreak", "rationale": "Rationale"}'
                        ),
                    )
                    mock_scoring_target.side_effect = scorer_responses_for_one_turn * depth
                    mock_target.side_effect = [
                        get_prompt_response_with_content(f"First branch level {level} target response")
                        for level in range(depth)
                    ]
                    result = await tap_orchestrator.apply_attack_strategy_async()

                    assert len(tap_orchestrator._orchestrators) == 1

                    assert result is None

                    assert mock_red_teaming_target.call_count == depth
                    assert mock_target.call_count == depth
                    assert mock_scoring_target.call_count == depth * (2 if on_topic_checking_enabled else 1)

                    # 4 conversation turns per depth level and 3 system prompts
                    # (but the red teaming system prompt doesn't get repeated),
                    # scoring prompts are not stored as of now
                    assert (
                        len(tap_orchestrator._memory.get_all_prompt_pieces())
                        == (6 if on_topic_checking_enabled else 5) * depth + 1
                    )


@pytest.mark.asyncio
@pytest.mark.parametrize("depth", [1, 2, 3])
@pytest.mark.parametrize("on_topic_checking_enabled", [True, False])
async def test_apply_strategy_multiturn_success_in_last_turn(
    prompt_target: MockPromptTarget,
    red_teaming_target: MockPromptTarget,
    memory: MemoryInterface,
    scoring_target: MockPromptTarget,
    depth: int,
    on_topic_checking_enabled: bool,
):
    with patch.object(CentralMemory, "get_memory_instance", return_value=memory):
        tap_orchestrator = TreeOfAttacksWithPruningOrchestrator(
            red_teaming_chat=red_teaming_target,
            prompt_target=prompt_target,
            scoring_target=scoring_target,
            width=1,
            branching_factor=1,
            depth=depth,
            conversation_objective="conversation objective",
            on_topic_checking_enabled=on_topic_checking_enabled,
            verbose=False,
        )

        with patch.object(tap_orchestrator._red_teaming_chat, "send_prompt_async") as mock_red_teaming_target:
            with patch.object(tap_orchestrator._scoring_target, "send_prompt_async") as mock_scoring_target:
                with patch.object(tap_orchestrator._prompt_target, "send_prompt_async") as mock_target:
                    mock_red_teaming_target.side_effect = [
                        get_prompt_response_with_content(
                            content=f'{{"improvements": "First branch level {level} improvement response", '
                            f'"prompt": "First branch level {level} red teaming prompt"}}',
                        )
                        for level in range(depth)
                    ]
                    mock_scoring_target_return_values_except_last_iteration = []
                    if on_topic_checking_enabled:
                        mock_scoring_target_return_values_except_last_iteration.append(
                            get_prompt_response_with_content(
                                '{"score_value": "True", "description": "On topic", "rationale": "Rationale"}'
                            ),
                        )
                    mock_scoring_target_return_values_except_last_iteration.append(
                        get_prompt_response_with_content(
                            '{"score_value": "5", "description": "No jailbreak", "rationale": "Rationale"}'
                        ),
                    )
                    mock_scoring_target_return_values_last_iteration = []
                    if on_topic_checking_enabled:
                        mock_scoring_target_return_values_last_iteration.append(
                            get_prompt_response_with_content(
                                '{"score_value": "True", "description": "On topic", "rationale": "Rationale"}'
                            ),
                        )
                    mock_scoring_target_return_values_last_iteration.append(
                        get_prompt_response_with_content(
                            '{"score_value": "10", "description": "Jailbreak", "rationale": "Rationale"}'
                        ),
                    )
                    mock_scoring_target.side_effect = (
                        mock_scoring_target_return_values_except_last_iteration * (depth - 1)
                        + mock_scoring_target_return_values_last_iteration
                    )
                    mock_target.side_effect = [
                        get_prompt_response_with_content(f"First branch level {level} target response")
                        for level in range(depth)
                    ]
                    result = await tap_orchestrator.apply_attack_strategy_async()

                    assert len(tap_orchestrator._orchestrators) == 1

                    assert result is not None
                    assert not result.pruned
                    assert result.completed
                    assert result.score == (10 - 1) / 9  # normalized score
                    assert result.orchestrator_id == tap_orchestrator._orchestrators[0].get_identifier()["id"]
                    assert result.prompt_target_conversation_id is not None

                    assert mock_red_teaming_target.call_count == depth
                    assert mock_target.call_count == depth
                    assert mock_scoring_target.call_count == depth * (2 if on_topic_checking_enabled else 1)

                    # 4 conversation turns per depth level and 3 system prompts
                    # (but the red teaming system prompt doesn't get repeated),
                    # scoring prompts are not stored as of now
                    assert (
                        len(tap_orchestrator._memory.get_all_prompt_pieces())
                        == (6 if on_topic_checking_enabled else 5) * depth + 1
                    )
