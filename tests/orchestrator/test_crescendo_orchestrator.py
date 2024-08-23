# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from pathlib import Path
import pytest
import tempfile
from tests.mocks import MockPromptTarget
import random
from unittest.mock import AsyncMock, patch

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
def orchestrator(mock_target: MockPromptTarget) -> CrescendoOrchestrator:
    return CrescendoOrchestrator(
        conversation_objective="conversation_objective",
        prompt_target=mock_target,
        red_teaming_chat=mock_target,
        scoring_target=mock_target,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("rounds", list(range(1, 11)))
async def test_apply_crescendo_attack(mock_target: MockPromptTarget, rounds: int):

    scoring_target = AsyncMock()
    orchestrator = CrescendoOrchestrator(
        conversation_objective="conversation_objective",
        prompt_target=mock_target,
        red_teaming_chat=mock_target,
        scoring_target=scoring_target,
    )

    orchestrator._prompt_normalizer = AsyncMock()

    normalizer_responses = []
    scorer_responses = []
    for round in range(1, rounds + 1):

        eval_percentage = round * 10 if round < rounds else 100
        refusal_value = round % 2 == 0

        if refusal_value:
            red_team_response = (
                '{"generated_question": "Red Teaming Response", "last_response_summary": "Last Response Summary",'
                '"rationale_behind_jailbreak": "Rationale Behind Jailbreak"}'
            )
            refusal_converted_value = '{"value": "True", "description": "Refusal", "rationale": "Refusal_Rationale"}'

            red_teaming_response = PromptRequestResponse(
                request_pieces=[
                    PromptRequestPiece(
                        converted_value=red_team_response, role="assistant", original_value=red_team_response
                    )
                ]
            )
            prompt_target_response = PromptRequestResponse(
                request_pieces=[
                    PromptRequestPiece(
                        converted_value="Prompt Target Response",
                        role="assistant",
                        original_value="Prompt Target Response",
                    )
                ]
            )

            refusal_response = PromptRequestResponse(
                request_pieces=[
                    PromptRequestPiece(
                        converted_value=refusal_converted_value,
                        role="assistant",
                        original_value=refusal_converted_value,
                    )
                ]
            )

            normalizer_responses.extend([red_teaming_response, prompt_target_response])
            scorer_responses.extend([refusal_response])

        red_team_response = (
            '{"generated_question": "Red Teaming Response", "last_response_summary": "Last Response Summary",'
            '"rationale_behind_jailbreak": "Rationale Behind Jailbreak"}'
        )
        refusal_converted_value = '{"value": "False", "description": "Refusal", "rationale": "Refusal_Rationale"}'
        eval_converted_value = (
            '{{"value": "True", "description": "Eval", "rationale": "Eval_Rationale", "metadata": "{0}"}}'.format(
                eval_percentage
            )
        )

        red_teaming_response = PromptRequestResponse(
            request_pieces=[
                PromptRequestPiece(
                    converted_value=red_team_response, role="assistant", original_value=red_team_response
                )
            ]
        )
        prompt_target_response = PromptRequestResponse(
            request_pieces=[
                PromptRequestPiece(
                    converted_value="Prompt Target Response", role="assistant", original_value="Prompt Target Response"
                )
            ]
        )

        refusal_response = PromptRequestResponse(
            request_pieces=[
                PromptRequestPiece(
                    converted_value=refusal_converted_value, role="assistant", original_value=refusal_converted_value
                )
            ]
        )
        eval_response = PromptRequestResponse(
            request_pieces=[
                PromptRequestPiece(
                    converted_value=eval_converted_value, role="assistant", original_value=eval_converted_value
                )
            ]
        )

        normalizer_responses.extend([red_teaming_response, prompt_target_response])
        scorer_responses.extend([refusal_response, eval_response])

    orchestrator._prompt_normalizer.send_prompt_async = AsyncMock(side_effect=normalizer_responses)

    scoring_target.send_prompt_async = AsyncMock(side_effect=scorer_responses)

    eval_score = await orchestrator.apply_crescendo_attack_async()

    assert eval_score is not None
    assert eval_score.get_value() is True
    assert eval_score.score_value_description == "Eval"
    assert eval_score.score_rationale == "Eval_Rationale"
    assert eval_score.score_metadata == "100"
    assert orchestrator._prompt_normalizer.send_prompt_async.call_count == (rounds // 2 + rounds) * 2
    assert scoring_target.send_prompt_async.call_count == rounds * 2 + rounds // 2


@pytest.mark.asyncio
@pytest.mark.parametrize("rounds", list(range(1, 11)))
async def test_max_backtracks_occurred(
    orchestrator: CrescendoOrchestrator, true_eval_score: Score, did_refuse_score: Score, rounds: int
):

    for round_num in range(1, rounds + 1):
        with (
            patch.object(orchestrator, "_get_attack_prompt", AsyncMock(return_value="attack_prompt")),
            patch.object(orchestrator, "_send_prompt_async", AsyncMock(return_value="last_response")),
            patch.object(
                orchestrator, "_backtrack_memory", AsyncMock(return_value="new_conversation_id")
            ) as mock_backtrack_memory,
            patch.object(orchestrator.refusal_scorer, "score_text_async", AsyncMock(return_value=[did_refuse_score])),
            patch.object(
                orchestrator.eval_judge_true_false_scorer, "score_text_async", AsyncMock(return_value=[true_eval_score])
            ) as mock_eval_judge,
        ):

            max_rounds = round_num
            max_backtracks = random.randint(1, 10)

            await orchestrator.apply_crescendo_attack_async(max_rounds=max_rounds, max_backtracks=max_backtracks)

            assert mock_backtrack_memory.call_count == (max_rounds + max_backtracks - 1)
            assert orchestrator._prompt_target_conversation_id == "new_conversation_id"
            assert mock_eval_judge.call_count == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("rounds", list(range(1, 11)))
async def test_no_backtracks_occurred(
    orchestrator: CrescendoOrchestrator, true_eval_score: Score, did_not_refuse_score: Score, rounds: int
):
    for round_num in range(1, rounds + 1):
        with (
            patch.object(orchestrator, "_get_attack_prompt", AsyncMock(return_value="attack_prompt")),
            patch.object(orchestrator, "_send_prompt_async", AsyncMock(return_value="last_response")),
            patch.object(
                orchestrator, "_backtrack_memory", AsyncMock(return_value="new_conversation_id")
            ) as mock_backtrack_memory,
            patch.object(
                orchestrator.refusal_scorer, "score_text_async", AsyncMock(return_value=[did_not_refuse_score])
            ),
            patch.object(
                orchestrator.eval_judge_true_false_scorer, "score_text_async", AsyncMock(return_value=[true_eval_score])
            ) as mock_eval_judge,
        ):

            max_rounds = round_num
            max_backtracks = random.randint(1, 10)

            await orchestrator.apply_crescendo_attack_async(max_rounds=max_rounds, max_backtracks=max_backtracks)

            assert mock_backtrack_memory.call_count == 0
            assert orchestrator._prompt_target_conversation_id != "new_conversation_id"
            assert mock_eval_judge.call_count == max_rounds


@pytest.mark.asyncio
async def test_value_error_exceptions(
    orchestrator: CrescendoOrchestrator, true_eval_score: Score, did_refuse_score: Score
):
    with (
        patch.object(orchestrator, "_get_attack_prompt", AsyncMock(return_value="attack_prompt")),
        patch.object(orchestrator, "_send_prompt_async", AsyncMock(return_value="last_response")),
        patch.object(orchestrator, "_backtrack_memory", AsyncMock(return_value="new_conversation_id")),
        patch.object(orchestrator.refusal_scorer, "score_text_async", AsyncMock(return_value=[did_refuse_score])),
        patch.object(
            orchestrator.eval_judge_true_false_scorer, "score_text_async", AsyncMock(return_value=[true_eval_score])
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
async def test_invalid_json_exceptions(orchestrator: CrescendoOrchestrator, red_team_response: str):

    red_teaming_return_value = PromptRequestResponse(
        request_pieces=[
            PromptRequestPiece(converted_value=red_team_response, role="assistant", original_value=red_team_response)
        ]
    )
    with (
        patch.object(
            orchestrator, "_backtrack_memory", AsyncMock(return_value="new_conversation_id")
        ) as mock_backtrack_memory,
        patch.object(orchestrator, "_prompt_normalizer", AsyncMock()) as mock_prompt_normalizer,
    ):

        mock_prompt_normalizer.send_prompt_async = AsyncMock(return_value=red_teaming_return_value)

        with pytest.raises(InvalidJsonException):
            await orchestrator._get_attack_prompt(round_num=1, eval_score=None, last_response=None)
            assert mock_backtrack_memory.call_count == 1
