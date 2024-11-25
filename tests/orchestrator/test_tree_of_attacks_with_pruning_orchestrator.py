# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import pytest

from typing import Generator
from unittest.mock import AsyncMock, MagicMock, patch

from pyrit.memory import MemoryInterface
from pyrit.memory import CentralMemory
from pyrit.models import PromptRequestResponse, PromptRequestPiece
from pyrit.orchestrator import TreeOfAttacksWithPruningOrchestrator
from tests.mocks import MockPromptTarget, get_memory_interface


@pytest.fixture
def memory() -> Generator[MemoryInterface, None, None]:
    yield from get_memory_interface()


@pytest.fixture
def objective_target(memory) -> MockPromptTarget:
    with patch.object(CentralMemory, "get_memory_instance", return_value=memory):
        return MockPromptTarget()


@pytest.fixture
def adversarial_chat(memory) -> MockPromptTarget:
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
        ],
    )


def test_invalid_width():
    with pytest.raises(ValueError) as e:
        TreeOfAttacksWithPruningOrchestrator(
            objective_target=MagicMock(),
            adversarial_chat=MagicMock(),
            scoring_target=MagicMock(),
            width=0,
        )
    assert e.match("The width of the tree must be at least 1.")


def test_invalid_branching_factor():
    with pytest.raises(ValueError) as e:
        TreeOfAttacksWithPruningOrchestrator(
            objective_target=MagicMock(),
            adversarial_chat=MagicMock(),
            scoring_target=MagicMock(),
            branching_factor=0,
        )
    assert e.match("The branching factor of the tree must be at least 1.")


def test_invalid_depth():
    with pytest.raises(ValueError) as e:
        TreeOfAttacksWithPruningOrchestrator(
            objective_target=MagicMock(),
            adversarial_chat=MagicMock(),
            scoring_target=MagicMock(),
            depth=0,
        )
    assert e.match("The depth of the tree must be at least 1.")


@pytest.mark.asyncio
@pytest.mark.parametrize("on_topic_checking_enabled", [True, False])
async def test_run_attack_single_turn_success(
    objective_target: MockPromptTarget,
    adversarial_chat: MockPromptTarget,
    scoring_target: MockPromptTarget,
    on_topic_checking_enabled: bool,
):
    tap_orchestrator = TreeOfAttacksWithPruningOrchestrator(
        adversarial_chat=adversarial_chat,
        objective_target=objective_target,
        scoring_target=scoring_target,
        width=1,
        branching_factor=1,
        depth=1,
        on_topic_checking_enabled=on_topic_checking_enabled,
        verbose=False,
    )

    with (
        patch.object(tap_orchestrator._adversarial_chat, "send_prompt_async") as mock_adversarial_chat,
        patch.object(tap_orchestrator._scoring_target, "send_prompt_async") as mock_scoring_target,
        patch.object(tap_orchestrator._objective_target, "send_prompt_async") as mock_objective_target,
    ):
        mock_adversarial_chat.return_value = get_prompt_response_with_content(
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
        mock_objective_target.return_value = expected_target_response

        result = await tap_orchestrator.run_attack_async(objective="objective")

        assert result.achieved_objective
        assert result.objective == "objective"

        assert "pruned" not in str(result.tree_visualization).lower()
        assert "score: 10/10" in str(result.tree_visualization).lower()

        mock_adversarial_chat.assert_called_once()
        mock_objective_target.assert_called_once()
        assert mock_scoring_target.call_count == 2 if on_topic_checking_enabled else 1

        # 4 conversation turns and 3 system prompts, scoring prompts are not stored as of now
        assert len(tap_orchestrator._memory.get_all_prompt_pieces()) == 7 if on_topic_checking_enabled else 6


@pytest.mark.asyncio
@pytest.mark.parametrize("on_topic_checking_enabled", [True, False])
async def test_run_attack_max_depth_reached(
    objective_target: MockPromptTarget,
    adversarial_chat: MockPromptTarget,
    scoring_target: MockPromptTarget,
    on_topic_checking_enabled: bool,
):

    tap_orchestrator = TreeOfAttacksWithPruningOrchestrator(
        adversarial_chat=adversarial_chat,
        objective_target=objective_target,
        scoring_target=scoring_target,
        width=1,
        branching_factor=1,
        depth=1,
        on_topic_checking_enabled=on_topic_checking_enabled,
        verbose=False,
    )

    with (
        patch.object(tap_orchestrator._adversarial_chat, "send_prompt_async") as mock_adversarial_chat,
        patch.object(tap_orchestrator._scoring_target, "send_prompt_async") as mock_scoring_target,
        patch.object(tap_orchestrator._objective_target, "send_prompt_async") as mock_objective_target,
    ):
        mock_adversarial_chat.return_value = get_prompt_response_with_content(
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
        mock_objective_target.return_value = expected_target_response

        result = await tap_orchestrator.run_attack_async(objective="objective")

        assert not result.achieved_objective
        assert result.objective == "objective"

        mock_adversarial_chat.assert_called_once()
        mock_objective_target.assert_called_once()
        assert mock_scoring_target.call_count == 2 if on_topic_checking_enabled else 1

        # 4 conversation turns and 3 system prompts, scoring prompts are not stored as of now
        assert len(tap_orchestrator._memory.get_all_prompt_pieces()) == 7 if on_topic_checking_enabled else 6


@pytest.mark.asyncio
@pytest.mark.parametrize("depth", [1, 2, 3])
async def test_run_attack_multiturn_failure(
    objective_target: MockPromptTarget,
    adversarial_chat: MockPromptTarget,
    scoring_target: MockPromptTarget,
    depth: int,
):
    tap_orchestrator = TreeOfAttacksWithPruningOrchestrator(
        adversarial_chat=adversarial_chat,
        objective_target=objective_target,
        scoring_target=scoring_target,
        width=1,
        branching_factor=1,
        depth=depth,
        on_topic_checking_enabled=False,
        verbose=False,
    )

    with patch.object(tap_orchestrator, "_send_prompt_to_nodes_async", new_callable=AsyncMock) as mock_send:

        async def side_effect(objective, nodes, tree_visualization):
            for node in nodes:
                node.completed = True
                node.score = 0.0

        mock_send.side_effect = side_effect

        result = await tap_orchestrator.run_attack_async(objective="objective")

        assert not result.achieved_objective
        assert mock_send.call_count == depth


@pytest.mark.asyncio
@pytest.mark.parametrize("depth", [1, 2, 3])
async def test_run_attack_success_in_last_turn(
    objective_target: MockPromptTarget,
    adversarial_chat: MockPromptTarget,
    scoring_target: MockPromptTarget,
    depth: int,
):
    tap_orchestrator = TreeOfAttacksWithPruningOrchestrator(
        adversarial_chat=adversarial_chat,
        objective_target=objective_target,
        scoring_target=scoring_target,
        width=1,
        branching_factor=1,
        depth=depth,
    )

    with patch.object(tap_orchestrator, "_send_prompt_to_nodes_async", new_callable=AsyncMock) as mock_send:

        call_count = 0

        async def side_effect(objective, nodes, tree_visualization):
            score = 0.0
            nonlocal call_count

            # score is 1.0 on last turn
            if call_count == depth - 1:
                score = 1.0

            call_count += 1

            for node in nodes:
                node.completed = True
                node.score = score

        mock_send.side_effect = side_effect

        result = await tap_orchestrator.run_attack_async(objective="objective")

        assert result.achieved_objective
        assert mock_send.call_count == depth


def test_prune_nodes_over_width():
    tap_orchestrator = TreeOfAttacksWithPruningOrchestrator(
        adversarial_chat=MagicMock(),
        objective_target=MagicMock(),
        scoring_target=MagicMock(),
        objective_achieved_score_threshold=0.7,
        width=2,
        branching_factor=1,
        depth=1,
    )

    nodes = [
        MagicMock(node_id="1", score=0.0, completed=True, off_topic=False),
        MagicMock(node_id="2", score=1.0, completed=True, off_topic=False),
        MagicMock(node_id="3", score=0.0, completed=True, off_topic=False),
        MagicMock(node_id="4", score=0.1, completed=True, off_topic=False),
        MagicMock(node_id="5", score=0.5, completed=True, off_topic=False),
    ]

    tree = MagicMock()

    pruned_nodes = tap_orchestrator._prune_nodes_over_width(nodes=nodes, tree_visualization=tree)

    assert len(pruned_nodes) == 2
    assert pruned_nodes[0].node_id == "2", "Highest score should be first"
    assert pruned_nodes[1].node_id == "5", "Highest score should be first"
