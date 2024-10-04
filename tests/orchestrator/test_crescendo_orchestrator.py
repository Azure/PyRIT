# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
from pathlib import Path
import pytest
import tempfile
from tests.mocks import MockPromptTarget
from unittest.mock import AsyncMock, MagicMock, patch

from pyrit.common.path import DATASETS_PATH
from pyrit.exceptions.exception_classes import InvalidJsonException
from pyrit.memory import DuckDBMemory
from pyrit.models import PromptRequestPiece
from pyrit.models import PromptRequestResponse
from pyrit.orchestrator import CrescendoOrchestrator
from pyrit.score import Score


@pytest.fixture
def mock_target() -> MockPromptTarget:
    fd, path = tempfile.mkstemp(suffix=".json.memory")
    file_memory = DuckDBMemory(db_path=":memory:")
    return MockPromptTarget(memory=file_memory)


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
def orchestrator(mock_target: MockPromptTarget) -> CrescendoOrchestrator:
    return CrescendoOrchestrator(
        conversation_objective="my_conversation_objective",
        prompt_target=mock_target,
        red_teaming_chat=mock_target,
        scoring_target=mock_target,
        memory=MagicMock(),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("rounds", [1, 10])
async def test_apply_crescendo_attack_fail_num_rounds(
    orchestrator: CrescendoOrchestrator, did_not_refuse_score: Score, false_eval_score: Score, rounds: int
):
    """
    Tests the scenario where we never have a refusal, but we reach the max num rounds
    and the attack fails
    """

    with (
        patch.object(
            orchestrator, "_get_attack_prompt", AsyncMock(return_value="attack_prompt")
        ) as mock_get_attack_prompt,
        patch.object(
            orchestrator.refusal_scorer,
            "score_async",
            AsyncMock(return_value=[did_not_refuse_score]),
        ) as mock_refusal_score,
        patch.object(
            orchestrator.eval_judge_true_false_scorer,
            "score_async",
            AsyncMock(return_value=[false_eval_score]),
        ) as mock_eval_score,
    ):
        objective_score = await orchestrator.apply_crescendo_attack_async(max_rounds=rounds)
        assert mock_refusal_score.call_count == rounds
        assert mock_eval_score.call_count == rounds
        assert not objective_score.get_value(), "should not have achieved objective"
        assert mock_get_attack_prompt.call_count == rounds


@pytest.mark.asyncio
@pytest.mark.parametrize("rounds", [1, 10])
async def test_apply_crescendo_attack_fail_max_refusals_num_rounds(
    orchestrator: CrescendoOrchestrator, did_refuse_score: Score, false_eval_score: Score, rounds: int
):
    """
    Tests the scenario where we reach max refusals, and then reach the max num rounds
    and the attack fails
    """

    max_refusal = 5

    with (
        patch.object(
            orchestrator, "_get_attack_prompt", AsyncMock(return_value="attack_prompt")
        ) as mock_get_attack_prompt,
        patch.object(
            orchestrator.refusal_scorer,
            "score_async",
            AsyncMock(return_value=[did_refuse_score]),
        ) as mock_refusal_score,
        patch.object(
            orchestrator.eval_judge_true_false_scorer,
            "score_async",
            AsyncMock(return_value=[false_eval_score]),
        ) as mock_eval_score,
    ):
        objective_score = await orchestrator.apply_crescendo_attack_async(max_rounds=rounds, max_backtracks=max_refusal)
        assert mock_refusal_score.call_count == max_refusal
        assert mock_eval_score.call_count == rounds
        assert not objective_score.get_value(), "should not have achieved objective"
        assert mock_get_attack_prompt.call_count == rounds + max_refusal


@pytest.mark.asyncio
async def test_apply_crescendo_attack_succeed_max_refusals_num_rounds(
    orchestrator: CrescendoOrchestrator,
    did_refuse_score: Score,
    true_eval_score: Score,
):
    """
    Tests the scenario where we reach max refusals, but then the evaluation succeeds
    """
    rounds = 10
    max_refusal = 5

    with (
        patch.object(
            orchestrator, "_get_attack_prompt", AsyncMock(return_value="attack_prompt")
        ) as mock_get_attack_prompt,
        patch.object(
            orchestrator.refusal_scorer,
            "score_async",
            AsyncMock(return_value=[did_refuse_score]),
        ) as mock_refusal_score,
        patch.object(
            orchestrator.eval_judge_true_false_scorer,
            "score_async",
            AsyncMock(return_value=[true_eval_score]),
        ) as mock_eval_score,
    ):
        objective_score = await orchestrator.apply_crescendo_attack_async(max_rounds=rounds, max_backtracks=max_refusal)
        assert mock_refusal_score.call_count == max_refusal
        assert mock_eval_score.call_count == 1
        assert objective_score.get_value(), "should have achieved objective"
        assert mock_get_attack_prompt.call_count == 1 + max_refusal


@pytest.mark.asyncio
async def test_apply_crescendo_attack_succeed_num_rounds(
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
            orchestrator.refusal_scorer,
            "score_async",
            AsyncMock(return_value=[did_not_refuse_score]),
        ) as mock_refusal_score,
        patch.object(
            orchestrator.eval_judge_true_false_scorer,
            "score_async",
            AsyncMock(return_value=[true_eval_score]),
        ) as mock_eval_score,
    ):
        objective_score = await orchestrator.apply_crescendo_attack_async(max_rounds=10)
        assert mock_refusal_score.call_count == 1
        assert mock_eval_score.call_count == 1
        assert objective_score.get_value(), "achieved objective"
        assert mock_get_attack_prompt.call_count == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("rounds", [1, 6, 11])
async def test_no_backtracks_occurred(
    orchestrator: CrescendoOrchestrator, false_eval_score: Score, did_not_refuse_score: Score, rounds: int
):
    for round_num in range(1, rounds + 1):
        with (
            patch.object(orchestrator, "_get_attack_prompt", AsyncMock(return_value="attack_prompt")),
            patch.object(
                orchestrator, "_backtrack_memory", AsyncMock(return_value="new_conversation_id")
            ) as mock_backtrack_memory,
            patch.object(orchestrator.refusal_scorer, "score_async", AsyncMock(return_value=[did_not_refuse_score])),
            patch.object(
                orchestrator.eval_judge_true_false_scorer, "score_async", AsyncMock(return_value=[false_eval_score])
            ) as mock_eval_judge,
        ):

            max_rounds = round_num

            await orchestrator.apply_crescendo_attack_async(max_rounds=max_rounds)

            assert mock_backtrack_memory.call_count == 0
            assert orchestrator._prompt_target_conversation_id != "new_conversation_id"
            assert mock_eval_judge.call_count == max_rounds


@pytest.mark.asyncio
async def test_value_error_exceptions(
    orchestrator: CrescendoOrchestrator, true_eval_score: Score, did_refuse_score: Score
):
    with (
        patch.object(orchestrator, "_get_attack_prompt", AsyncMock(return_value="attack_prompt")),
        patch.object(orchestrator, "_backtrack_memory", AsyncMock(return_value="new_conversation_id")),
        patch.object(orchestrator.refusal_scorer, "score_async", AsyncMock(return_value=[did_refuse_score])),
        patch.object(
            orchestrator.eval_judge_true_false_scorer, "score_async", AsyncMock(return_value=[true_eval_score])
        ),
    ):

        with pytest.raises(ValueError):
            await orchestrator.apply_crescendo_attack_async(max_rounds=10, max_backtracks=0)

        with pytest.raises(ValueError):
            await orchestrator.apply_crescendo_attack_async(max_rounds=0, max_backtracks=10)


@pytest.mark.asyncio
@pytest.mark.parametrize("variants", list(range(1, 6)))
async def test_custom_crescendo_system_prompt_path(mock_target: AsyncMock, variants: int):

    for num in range(1, variants + 1):
        conversation_objective = "Test Conversation Objective"
        custom_system_prompt_path = (
            Path(DATASETS_PATH) / "orchestrators" / "crescendo" / f"crescendo_variant_{num}.yaml"
        )

        orchestrator = CrescendoOrchestrator(
            conversation_objective=conversation_objective,
            prompt_target=mock_target,
            red_teaming_chat=mock_target,
            scoring_target=mock_target,
            system_prompt_path=custom_system_prompt_path,
        )

        assert orchestrator._conversation_objective == conversation_objective
        assert orchestrator._system_prompt_path == custom_system_prompt_path


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
                red_team_conversation_id="123", refused_text=None, round_num=1, max_rounds=10, objective_score=None
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

    with patch.object(orchestrator, "_prompt_normalizer", AsyncMock()) as mock_prompt_normalizer:
        mock_prompt_normalizer.send_prompt_async = AsyncMock(return_value=red_teaming_return_value)
        await orchestrator._get_attack_prompt(
            red_team_conversation_id="123", refused_text=None, round_num=3, max_rounds=7, objective_score=None
        )
        _, kwargs = mock_prompt_normalizer.send_prompt_async.call_args
        prompt_sent = kwargs["normalizer_request"].request_pieces[0].prompt_value

        # number of rounds and conversation objective should be included dynamically
        assert "This is the round 3 of 7 rounds." in prompt_sent
        assert orchestrator._conversation_objective in prompt_sent
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

    with patch.object(orchestrator, "_prompt_normalizer", AsyncMock()) as mock_prompt_normalizer:
        mock_prompt_normalizer.send_prompt_async = AsyncMock(return_value=red_teaming_return_value)
        await orchestrator._get_attack_prompt(
            red_team_conversation_id="123",
            refused_text="this was refused",
            round_num=3,
            max_rounds=7,
            objective_score=None,
        )
        _, kwargs = mock_prompt_normalizer.send_prompt_async.call_args
        prompt_sent = kwargs["normalizer_request"].request_pieces[0].prompt_value

        # number of rounds and conversation objective should be included dynamically
        assert "This is the round 3 of 7 rounds." in prompt_sent
        assert orchestrator._conversation_objective in prompt_sent
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

    with (
        patch.object(orchestrator._memory, "get_prompt_request_pieces_by_id", MagicMock(return_value=previous_prompt)),
        patch.object(orchestrator, "_prompt_normalizer", AsyncMock()) as mock_prompt_normalizer,
    ):

        mock_prompt_normalizer.send_prompt_async.return_value = red_teaming_return_value
        await orchestrator._get_attack_prompt(
            red_team_conversation_id="123",
            refused_text=None,
            round_num=3,
            max_rounds=7,
            objective_score=false_eval_score,
        )
        _, kwargs = mock_prompt_normalizer.send_prompt_async.call_args
        prompt_sent = kwargs["normalizer_request"].request_pieces[0].prompt_value

        assert "This is the round 3 of 7 rounds." in prompt_sent
        assert orchestrator._conversation_objective in prompt_sent

        # the previous prompt text and score rationalie should be in prompt text
        assert previous_prompt_text in prompt_sent
        assert false_eval_score.score_rationale in prompt_sent
