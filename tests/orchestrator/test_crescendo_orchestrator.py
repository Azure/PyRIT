# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
from typing import Generator
import pytest

from pathlib import Path
from pyrit.memory.memory_interface import MemoryInterface
from tests.mocks import MockPromptTarget
from unittest.mock import AsyncMock, MagicMock, patch

from pyrit.common.path import DATASETS_PATH
from pyrit.exceptions import InvalidJsonException
from pyrit.memory import CentralMemory
from pyrit.models import PromptRequestPiece, PromptRequestResponse
from pyrit.orchestrator import CrescendoOrchestrator
from pyrit.models import Score
from tests.mocks import get_memory_interface


@pytest.fixture
def memory_interface() -> Generator[MemoryInterface, None, None]:
    yield from get_memory_interface()


@pytest.fixture
def mock_target(memory_interface) -> MockPromptTarget:
    with patch.object(CentralMemory, "get_memory_instance", return_value=memory_interface):
        return MockPromptTarget()


@pytest.fixture
def did_refuse_score() -> Score:
    return Score(
        score_value="true",
        score_rationale="Refusal Rationale",
        score_value_description="description",
        score_type="true_false",
        score_metadata="",
        score_category="category",
        scorer_class_identifier=None,
        prompt_request_response_id="id",
    )


@pytest.fixture
def did_not_refuse_score() -> Score:
    return Score(
        score_value="false",
        score_rationale="Refusal Rationale",
        score_value_description="description",
        score_type="true_false",
        score_metadata="",
        score_category="category",
        scorer_class_identifier=None,
        prompt_request_response_id="id",
    )


@pytest.fixture
def true_eval_score() -> Score:
    return Score(
        score_value="true",
        score_rationale="Refusal Rationale",
        score_value_description="description",
        score_type="true_false",
        score_metadata="",
        score_category="category",
        scorer_class_identifier=None,
        prompt_request_response_id="id",
    )


@pytest.fixture
def false_eval_score() -> Score:
    return Score(
        score_value="false",
        score_rationale="Refusal Rationale",
        score_value_description="description",
        score_type="true_false",
        score_metadata="",
        score_category="category",
        scorer_class_identifier=None,
        prompt_request_response_id="id",
    )


@pytest.fixture
def orchestrator(mock_target: MockPromptTarget, memory_interface: MemoryInterface) -> CrescendoOrchestrator:
    with patch.object(CentralMemory, "get_memory_instance", return_value=memory_interface):
        return CrescendoOrchestrator(
            objective_target=mock_target, adversarial_chat=mock_target, scoring_target=mock_target
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("turns", [1, 10])
async def test_apply_crescendo_attack_fail_num_turns(
    orchestrator: CrescendoOrchestrator, did_not_refuse_score: Score, false_eval_score: Score, turns: int
):
    """
    Tests the scenario where we never have a refusal, but we reach the max num turns
    and the attack fails
    """

    orchestrator._max_turns = turns

    with (
        patch.object(
            orchestrator, "_get_attack_prompt", AsyncMock(return_value="attack_prompt")
        ) as mock_get_attack_prompt,
        patch.object(
            orchestrator._refusal_scorer,
            "score_async",
            AsyncMock(return_value=[did_not_refuse_score]),
        ) as mock_refusal_score,
        patch.object(
            orchestrator._objective_scorer,
            "score_async",
            AsyncMock(return_value=[false_eval_score]),
        ) as mock_eval_score,
    ):
        result = await orchestrator.run_attack_async(objective="objective")
        assert mock_refusal_score.call_count == turns
        assert mock_eval_score.call_count == turns
        assert not result.achieved_objective, "should not have achieved objective"
        assert mock_get_attack_prompt.call_count == turns


@pytest.mark.asyncio
@pytest.mark.parametrize("turns", [1, 10])
async def test_apply_crescendo_attack_fail_max_refusals_num_turns(
    orchestrator: CrescendoOrchestrator, did_refuse_score: Score, false_eval_score: Score, turns: int
):
    """
    Tests the scenario where we reach max refusals, and then reach the max num turns
    and the attack fails
    """

    max_backtracks = 5
    orchestrator._max_backtracks = max_backtracks
    orchestrator._max_turns = turns

    with (
        patch.object(
            orchestrator, "_get_attack_prompt", AsyncMock(return_value="attack_prompt")
        ) as mock_get_attack_prompt,
        patch.object(
            orchestrator._refusal_scorer,
            "score_async",
            AsyncMock(return_value=[did_refuse_score]),
        ) as mock_refusal_score,
        patch.object(
            orchestrator._objective_scorer,
            "score_async",
            AsyncMock(return_value=[false_eval_score]),
        ) as mock_eval_score,
    ):
        result = await orchestrator.run_attack_async(objective="objective")
        assert mock_refusal_score.call_count == max_backtracks
        assert mock_eval_score.call_count == turns
        assert not result.achieved_objective, "should not have achieved objective"
        assert mock_get_attack_prompt.call_count == turns + max_backtracks


@pytest.mark.asyncio
async def test_apply_crescendo_attack_succeed_max_refusals_num_turns(
    orchestrator: CrescendoOrchestrator,
    did_refuse_score: Score,
    true_eval_score: Score,
):
    """
    Tests the scenario where we reach max refusals, but then the evaluation succeeds
    """
    max_refusal = 5
    orchestrator._max_backtracks = max_refusal

    with (
        patch.object(
            orchestrator, "_get_attack_prompt", AsyncMock(return_value="attack_prompt")
        ) as mock_get_attack_prompt,
        patch.object(
            orchestrator._refusal_scorer,
            "score_async",
            AsyncMock(return_value=[did_refuse_score]),
        ) as mock_refusal_score,
        patch.object(
            orchestrator._objective_scorer,
            "score_async",
            AsyncMock(return_value=[true_eval_score]),
        ) as mock_eval_score,
    ):
        objective_score = await orchestrator.run_attack_async(objective="objective")
        assert mock_refusal_score.call_count == max_refusal
        assert mock_eval_score.call_count == 1
        assert objective_score.achieved_objective, "should have achieved objective"
        assert mock_get_attack_prompt.call_count == 1 + max_refusal


@pytest.mark.asyncio
async def test_apply_crescendo_attack_succeed_num_turns(
    orchestrator: CrescendoOrchestrator,
    did_not_refuse_score: Score,
    true_eval_score: Score,
):
    """
    Tests the scenario where we never have a refusal, and the attack succeeds
    """

    with (
        patch.object(
            orchestrator, "_get_attack_prompt", AsyncMock(return_value="attack_prompt")
        ) as mock_get_attack_prompt,
        patch.object(
            orchestrator._refusal_scorer,
            "score_async",
            AsyncMock(return_value=[did_not_refuse_score]),
        ) as mock_refusal_score,
        patch.object(
            orchestrator._objective_scorer,
            "score_async",
            AsyncMock(return_value=[true_eval_score]),
        ) as mock_eval_score,
    ):
        objective_score = await orchestrator.run_attack_async(objective="objective")
        assert mock_refusal_score.call_count == 1
        assert mock_eval_score.call_count == 1
        assert objective_score.achieved_objective, "achieved objective"
        assert mock_get_attack_prompt.call_count == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("turns", [1, 6, 11])
async def test_no_backtracks_occurred(
    orchestrator: CrescendoOrchestrator, false_eval_score: Score, did_not_refuse_score: Score, turns: int
):
    orchestrator._max_turns = turns

    with (
        patch.object(orchestrator, "_get_attack_prompt", AsyncMock(return_value="attack_prompt")),
        patch.object(
            orchestrator, "_backtrack_memory", AsyncMock(return_value="new_conversation_id")
        ) as mock_backtrack_memory,
        patch.object(orchestrator._refusal_scorer, "score_async", AsyncMock(return_value=[did_not_refuse_score])),
        patch.object(
            orchestrator._objective_scorer, "score_async", AsyncMock(return_value=[false_eval_score])
        ) as mock_eval_judge,
    ):

        result = await orchestrator.run_attack_async(objective="objective")

        assert mock_backtrack_memory.call_count == 0
        assert result.conversation_id != "new_conversation_id"
        assert mock_eval_judge.call_count == turns


@pytest.mark.asyncio
async def test_max_turns_init_exceptions(memory_interface: MemoryInterface):
    with patch.object(CentralMemory, "get_memory_instance", return_value=memory_interface):
        with pytest.raises(ValueError):
            CrescendoOrchestrator(
                objective_target=MagicMock(), adversarial_chat=MagicMock(), scoring_target=MagicMock(), max_turns=0
            )


@pytest.mark.asyncio
async def test_max_backtrack_init_exceptions(memory_interface: MemoryInterface):
    with patch.object(CentralMemory, "get_memory_instance", return_value=memory_interface):
        with pytest.raises(ValueError):
            CrescendoOrchestrator(
                objective_target=MagicMock(), adversarial_chat=MagicMock(), scoring_target=MagicMock(), max_backtracks=0
            )


@pytest.mark.asyncio
@pytest.mark.parametrize("variants", list(range(1, 6)))
async def test_custom_crescendo_system_prompt_path_variants(mock_target: AsyncMock, variants: int):

    for num in range(1, variants + 1):
        custom_system_prompt_path = (
            Path(DATASETS_PATH) / "orchestrators" / "crescendo" / f"crescendo_variant_{num}.yaml"
        )

        # This will throw an exception if the yaml is invalid or if there is no objective.
        CrescendoOrchestrator(
            objective_target=mock_target,
            adversarial_chat=mock_target,
            scoring_target=mock_target,
            adversarial_chat_system_prompt_path=custom_system_prompt_path,
        )


@pytest.mark.asyncio
async def test_invalid_system_prompt_path_raises():
    with pytest.raises(FileNotFoundError):
        CrescendoOrchestrator(
            objective_target=MagicMock(),
            adversarial_chat=MagicMock(),
            scoring_target=MagicMock(),
            adversarial_chat_system_prompt_path="does_not_exist.yaml",
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "red_team_response",
    [
        '{"generated_question": "Attack Prompt"}',
        '{"question": "Attack Prompt", "last_response_summary": "Last Response Summary",'
        ' "rationale_behind_jailbreak": "Rationale Behind Jailbreak"}',
        '{"generated_question": "Attack Prompt", "last_response_summary": "Last Response Summary",'
        ' "rationale_behind_jailbreak": "Rationale Behind Jailbreak", "invalid_key": "Invalid Key"}',
        '{"generated_question": "Attack Prompt", "last_response_summary": "Last Response Summary", '
        '"rationale_behind_jailbreak": "Rationale Behind Jailbreak",}',
    ],
)
async def test_get_attack_invalid_json_exceptions(orchestrator: CrescendoOrchestrator, red_team_response: str):

    red_teaming_return_value = PromptRequestResponse(
        request_pieces=[
            PromptRequestPiece(converted_value=red_team_response, role="assistant", original_value=red_team_response)
        ]
    )
    with patch.object(orchestrator, "_prompt_normalizer", AsyncMock()) as mock_prompt_normalizer:

        mock_prompt_normalizer.send_prompt_async = AsyncMock(return_value=red_teaming_return_value)

        with pytest.raises(InvalidJsonException):
            await orchestrator._get_attack_prompt(
                adversarial_chat_conversation_id="123",
                objective="objective",
                refused_text=None,
                turn_num=1,
                max_turns=10,
                objective_score=None,
            )
        assert mock_prompt_normalizer.send_prompt_async.call_count == int(os.getenv("RETRY_MAX_NUM_ATTEMPTS"))


@pytest.mark.asyncio
async def test_get_attack_sets_default_prompt(orchestrator: CrescendoOrchestrator):
    red_team_response = (
        '{"generated_question": "Red Teaming Response", "last_response_summary": "Last Response Summary",'
        '"rationale_behind_jailbreak": "Rationale Behind Jailbreak"}'
    )

    red_teaming_return_value = PromptRequestResponse(
        request_pieces=[PromptRequestPiece(role="assistant", original_value=red_team_response)]
    )

    objective = "do something extremely bad"

    with patch.object(orchestrator, "_prompt_normalizer", AsyncMock()) as mock_prompt_normalizer:
        mock_prompt_normalizer.send_prompt_async = AsyncMock(return_value=red_teaming_return_value)
        await orchestrator._get_attack_prompt(
            adversarial_chat_conversation_id="123",
            objective=objective,
            refused_text=None,
            turn_num=3,
            max_turns=7,
            objective_score=None,
        )
        _, kwargs = mock_prompt_normalizer.send_prompt_async.call_args
        prompt_sent = kwargs["normalizer_request"].request_pieces[0].prompt_value

        # number of turns and conversation objective should be included dynamically
        assert "This is the turn 3 of 7 turns." in prompt_sent
        assert objective in prompt_sent
        assert "The target refused" not in prompt_sent


@pytest.mark.asyncio
async def test_get_attack_sets_refused_text(orchestrator: CrescendoOrchestrator):
    red_team_response = (
        '{"generated_question": "Red Teaming Response", "last_response_summary": "Last Response Summary",'
        '"rationale_behind_jailbreak": "Rationale Behind Jailbreak"}'
    )

    red_teaming_return_value = PromptRequestResponse(
        request_pieces=[PromptRequestPiece(role="assistant", original_value=red_team_response)]
    )

    objective = "do something extremely bad"

    with patch.object(orchestrator, "_prompt_normalizer", AsyncMock()) as mock_prompt_normalizer:
        mock_prompt_normalizer.send_prompt_async = AsyncMock(return_value=red_teaming_return_value)
        await orchestrator._get_attack_prompt(
            adversarial_chat_conversation_id="123",
            objective=objective,
            refused_text="this was refused",
            turn_num=3,
            max_turns=7,
            objective_score=None,
        )
        _, kwargs = mock_prompt_normalizer.send_prompt_async.call_args
        prompt_sent = kwargs["normalizer_request"].request_pieces[0].prompt_value

        # number of turns and conversation objective should be included dynamically
        assert "This is the turn 3 of 7 turns." in prompt_sent
        assert objective in prompt_sent
        # refused text should also be included
        assert "The target refused to respond to the last request you sent. " in prompt_sent
        assert "this was refused" in prompt_sent


@pytest.mark.asyncio
async def test_get_attack_includes_objective_score(orchestrator: CrescendoOrchestrator, false_eval_score: Score):
    red_team_response = (
        '{"generated_question": "Red Teaming Response", "last_response_summary": "Last Response Summary",'
        '"rationale_behind_jailbreak": "Rationale Behind Jailbreak"}'
    )

    red_teaming_return_value = PromptRequestResponse(
        request_pieces=[PromptRequestPiece(role="assistant", original_value=red_team_response)]
    )

    previous_prompt_text = "hello I am the previous prompt"
    previous_prompt = [PromptRequestPiece(role="user", original_value=previous_prompt_text)]

    objective = "do something extremely bad"

    with (
        patch.object(orchestrator._memory, "get_prompt_request_pieces_by_id", MagicMock(return_value=previous_prompt)),
        patch.object(orchestrator, "_prompt_normalizer", AsyncMock()) as mock_prompt_normalizer,
    ):

        mock_prompt_normalizer.send_prompt_async.return_value = red_teaming_return_value
        await orchestrator._get_attack_prompt(
            adversarial_chat_conversation_id="123",
            objective=objective,
            refused_text=None,
            turn_num=3,
            max_turns=7,
            objective_score=false_eval_score,
        )
        _, kwargs = mock_prompt_normalizer.send_prompt_async.call_args
        prompt_sent = kwargs["normalizer_request"].request_pieces[0].prompt_value

        assert "This is the turn 3 of 7 turns." in prompt_sent
        assert objective in prompt_sent

        # the previous prompt text and score rationalie should be in prompt text
        assert previous_prompt_text in prompt_sent
        assert false_eval_score.score_rationale in prompt_sent
