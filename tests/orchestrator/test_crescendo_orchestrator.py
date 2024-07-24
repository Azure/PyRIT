# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from pathlib import Path
import pytest
import tempfile
from tests.mocks import MockPromptTarget
import random
from unittest.mock import AsyncMock, Mock

from pyrit.common.path import DATASETS_PATH
from pyrit.memory import DuckDBMemory
from pyrit.models import PromptRequestPiece
from pyrit.models.prompt_request_response import PromptRequestResponse
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
            prompt_target_respose = PromptRequestResponse(
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

            normalizer_responses.extend([red_teaming_response, prompt_target_respose])
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
        prompt_target_respose = PromptRequestResponse(
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
                    converted_value=eval_converted_value, role="assistant", original_value=refusal_converted_value
                )
            ]
        )

        normalizer_responses.extend([red_teaming_response, prompt_target_respose])
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
async def test_max_backtracks_occured(
    orchestrator: CrescendoOrchestrator, true_eval_score: Score, did_refuse_score: Score, rounds: int
):
    for round in range(1, rounds + 1):
        orchestrator._get_attack_prompt = AsyncMock(return_value="attack_prompt")
        orchestrator._send_prompt_async = AsyncMock(return_value="last_response")
        orchestrator._backtrack_memory = AsyncMock(return_value="new_conversation_id")

        orchestrator.refusal_scorer.score_text_async = AsyncMock(return_value=[did_refuse_score])
        orchestrator.eval_judge_true_false_scorer.score_text_async = AsyncMock(return_value=[true_eval_score])
        orchestrator.print_target_memory = Mock(return_value=None)

        max_rounds = round
        max_backtracks = random.randint(1, 10)

        await orchestrator.apply_crescendo_attack_async(max_rounds=max_rounds, max_backtracks=max_backtracks)

    assert orchestrator._backtrack_memory.call_count == int(max_rounds + max_backtracks - 1)
    assert orchestrator._prompt_target_conversation_id == "new_conversation_id"
    assert orchestrator.eval_judge_true_false_scorer.score_text_async.call_count == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("rounds", list(range(1, 11)))
async def test_no_backtracks_occured(
    orchestrator: CrescendoOrchestrator, true_eval_score: Score, did_not_refuse_score: Score, rounds: int
):
    for round in range(1, rounds + 1):
        orchestrator._get_attack_prompt = AsyncMock(return_value="attack_prompt")
        orchestrator._send_prompt_async = AsyncMock(return_value="last_response")
        orchestrator._backtrack_memory = AsyncMock(return_value="new_conversation_id")

        orchestrator.refusal_scorer.score_text_async = AsyncMock(return_value=[did_not_refuse_score])
        orchestrator.eval_judge_true_false_scorer.score_text_async = AsyncMock(return_value=[true_eval_score])
        orchestrator.print_target_memory = Mock(return_value=None)

        max_rounds = round
        max_backtracks = random.randint(1, 10)

        await orchestrator.apply_crescendo_attack_async(max_rounds=max_rounds, max_backtracks=max_backtracks)

    assert orchestrator._backtrack_memory.call_count == 0
    assert orchestrator._prompt_target_conversation_id != "new_conversation_id"
    assert orchestrator.eval_judge_true_false_scorer.score_text_async.call_count == max_rounds


@pytest.mark.asyncio
async def test_value_error_exceptions(
    orchestrator: CrescendoOrchestrator, true_eval_score: Score, did_refuse_score: Score
):
    orchestrator._get_attack_prompt = AsyncMock(return_value="attack_prompt")
    orchestrator._send_prompt_async = AsyncMock(return_value="last_response")
    orchestrator._backtrack_memory = AsyncMock(return_value="new_conversation_id")

    orchestrator.refusal_scorer.score_text_async = AsyncMock(return_value=[did_refuse_score])
    orchestrator.eval_judge_true_false_scorer.score_text_async = AsyncMock(return_value=[true_eval_score])
    orchestrator.print_target_memory = Mock(return_value=None)

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
@pytest.mark.parametrize("rounds", list(range(1, 11)))
async def test_get_attack_prompt(orchestrator: CrescendoOrchestrator, rounds: int, mock_target: AsyncMock):

    red_team_response = (
        '{"generated_question": "Red Teaming Response", "last_response_summary": "Last Response Summary",'
        '"rationale_behind_jailbreak": "Rationale Behind Jailbreak"}'
    )
    red_teaming_return_value = PromptRequestResponse(
        request_pieces=[
            PromptRequestPiece(converted_value=red_team_response, role="assistant", original_value=red_team_response)
        ]
    )
    orchestrator._prompt_normalizer.send_prompt_async = AsyncMock(return_value=red_teaming_return_value)
    orchestrator._create_normalizer_request = AsyncMock(return_value="This is first round")

    for round in range(1, rounds + 1):
        orchestrator._is_first_turn_with_red_teaming_chat = (
            AsyncMock(return_value=True) if round == 1 else AsyncMock(return_value=False)
        )

        if orchestrator._is_first_turn_with_red_teaming_chat():
            orchestrator._red_teaming_chat.set_system_prompt = AsyncMock()

        orchestrator._create_normalizer_request = (
            AsyncMock(return_value="This is first round")
            if round == 1
            else AsyncMock(return_value="This is first round")
        )

        orchestrator._prompt_normalizer.send_prompt_async(normalizer_request="This is first round")

        red_team_value = (
            '{"generated_question": "Attack Prompt", "last_response_summary": "Last Response Summary",'
            '"rationale_behind_jailbreak": "Rationale Behind Jailbreak"}'
        )
        red_teaming_response = PromptRequestResponse(
            request_pieces=[
                PromptRequestPiece(converted_value=red_team_value, role="assistant", original_value=red_team_value)
            ]
        )

    orchestrator._red_teaming_chat_conversation_id = "red_teaming_chat_conversation_id"
    orchestrator._system_prompt = "system_prompt"

    attack_prompt = await orchestrator._get_attack_prompt(round_num=1, eval_score=None, last_response=None)

    assert attack_prompt == "Attack Prompt"

    # orchestrator._is_first_turn_with_red_teaming_chat = AsyncMock(return_value=True)
    # orchestrator._create_normalizer_request = AsyncMock()
    # orchestrator._prompt_normalizer.send_prompt_async = AsyncMock(return_value=AsyncMock(request_pieces=[AsyncMock(converted_value="generated_question")]))

    # attack_prompt = await orchestrator._get_attack_prompt(round_num=1)

    # assert attack_prompt == "generated_question"
    # assert orchestrator._is_first_turn_with_red_teaming_chat.call_count == 1
    # assert orchestrator._create_normalizer_request.call_count == 1
    # assert orchestrator._prompt_normalizer.send_prompt_async.call_count == 1

    # PromptRequestPiece(
    #         request_pieces=[
    #             MockPromptRequestPiece(converted_value='{"generated_question": "Attack Prompt"}')
    #         ]
    #     )
    # )


async def test_send_prompt_async(orchestrator: CrescendoOrchestrator):

    orchestrator._create_normalizer_request = AsyncMock()
    orchestrator._prompt_normalizer.send_prompt_async = AsyncMock(
        return_value=AsyncMock(request_pieces=[AsyncMock(converted_value="last_response")])
    )

    last_response = await orchestrator._send_prompt_async(attack_prompt="attack_prompt")

    assert last_response == "last_response"
    assert orchestrator._create_normalizer_request.call_count == 1
    assert orchestrator._prompt_normalizer.send_prompt_async.call_count == 1


async def test_backtrack_memory(orchestrator: CrescendoOrchestrator):

    orchestrator._memory.duplicate_conversation_excluding_last_turn = AsyncMock(return_value="new_conversation_id")

    new_conversation_id = await orchestrator._backtrack_memory(conversation_id="conversation_id")

    assert new_conversation_id == "new_conversation_id"
    assert orchestrator._memory.duplicate_conversation_excluding_last_turn.call_count == 1


async def test_get_refusal_score_async(orchestrator: CrescendoOrchestrator):
    orchestrator._get_refusal_score_async = AsyncMock(return_value=(False, "refusal_rationale"))

    is_refusal, refusal_rationale = await orchestrator._get_refusal_score_async(
        attack_prompt="attack_prompt", last_response="last_response"
    )

    assert is_refusal is False
    assert refusal_rationale == "refusal_rationale"
    assert orchestrator._get_refusal_score_async.call_count == 1


async def test_get_eval_score_async(orchestrator: CrescendoOrchestrator):

    orchestrator._get_eval_score_async = AsyncMock(return_value=(True, 100))

    eval_score = await orchestrator._get_eval_score_async(last_response="last_response")

    assert eval_score is not None
    assert eval_score.get_value() is True
    assert eval_score.score_value_description == "Eval"
    assert eval_score.score_rationale == "Eval_Rationale"
    assert eval_score.score_metadata == "100"
    assert orchestrator._get_eval_score_async.call_count == 1
    pass


async def test_crescendo_get_attack_prompt_json_retry(mock_target: MockPromptTarget, rounds: int):
    pass


@pytest.mark.asyncio
async def test_get_refusal(mock_target: MockPromptTarget):
    scoring_target = AsyncMock()
    orchestrator = CrescendoOrchestrator(
        conversation_objective="conversation_objective",
        prompt_target=mock_target,
        red_teaming_chat=mock_target,
        scoring_target=scoring_target,
    )
    orchestrator.refusal_scorer = AsyncMock()

    attack_prompt = "attack prompt"
    last_response = "last response"
    expected_is_refusal = True
    expected_refusal_rationale = "Test rationale"

    mock_score_response = AsyncMock()
    mock_score_response.get_value.return_value = expected_is_refusal
    mock_score_response.score_rationale = expected_refusal_rationale

    orchestrator.refusal_scorer.score_text_async.return_value = [mock_score_response]

    is_refusal, refusal_rationale = await orchestrator._get_refusal_score_async(attack_prompt, last_response)

    assert is_refusal == expected_is_refusal
    assert refusal_rationale == expected_refusal_rationale
    orchestrator.refusal_scorer.score_text_async.assert_called_once()


async def test_crescendo_prompt_target_backtrack(
    orchestrator: CrescendoOrchestrator, mock_target: MockPromptTarget, rounds: int
):
    pass


async def test_crescendo_red_teaming_chat_backtrack(
    orchestrator: CrescendoOrchestrator, mock_target: MockPromptTarget, rounds: int
):
    pass


async def test_crescendo_get_eval(orchestrator: CrescendoOrchestrator, mock_target: MockPromptTarget, ounds: int):
    pass
