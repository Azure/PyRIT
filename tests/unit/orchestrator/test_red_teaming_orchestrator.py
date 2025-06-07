# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import copy
import json
import pathlib
from typing import Dict, List
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
from httpx import Response
from unit.mocks import openai_response_json_dict

from pyrit.common.path import DATASETS_PATH
from pyrit.memory import CentralMemory
from pyrit.models import PromptRequestPiece, PromptRequestResponse, Score
from pyrit.orchestrator import RedTeamingOrchestrator
from pyrit.prompt_target import OpenAIChatTarget, PromptTarget
from pyrit.score import Scorer


@pytest.fixture
def chat_completion_engine(patch_central_database) -> OpenAIChatTarget:
    return OpenAIChatTarget(model_name="test", endpoint="test", api_key="test")


@pytest.fixture
def prompt_target(patch_central_database) -> OpenAIChatTarget:
    return OpenAIChatTarget(model_name="test", endpoint="test", api_key="test")


@pytest.fixture
def red_team_system_prompt_path() -> pathlib.Path:
    return pathlib.Path(DATASETS_PATH) / "orchestrators" / "red_teaming" / "text_generation.yaml"


@pytest.fixture
def openai_response_json() -> dict:
    return openai_response_json_dict()


def _check_orchestrator_memory(memory, num_turns: int):
    conversations = memory.get_prompt_request_pieces()
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


def _get_http_response(text: str, response_dict: dict) -> Response:
    dict_copy = copy.deepcopy(response_dict)
    dict_copy["choices"][0]["message"]["content"] = text
    return Response(status_code=200, content=json.dumps(dict_copy))


@pytest.mark.asyncio
async def test_send_prompt_twice(
    prompt_target: PromptTarget,
    chat_completion_engine: OpenAIChatTarget,
    openai_response_json: dict,
):

    first_adversarial_response_http_response = _get_http_response("First adversarial response", openai_response_json)
    first_objective_response_http_response = _get_http_response("First objective response", openai_response_json)
    second_adversarial_response_http_response = _get_http_response("Second adversarial response", openai_response_json)
    second_objective_response_http_response = _get_http_response("Second objective response", openai_response_json)

    scorer = MagicMock(Scorer)
    scorer.scorer_type = "true_false"
    red_teaming_orchestrator = RedTeamingOrchestrator(
        adversarial_chat=chat_completion_engine,
        objective_target=prompt_target,
        objective_scorer=scorer,
    )

    prompt_target_conversation_id = str(uuid4())
    red_teaming_chat_conversation_id = str(uuid4())

    with patch("pyrit.common.net_utility.make_request_and_raise_if_error_async") as mock_make_request:
        mock_make_request.side_effect = [
            first_adversarial_response_http_response,
            first_objective_response_http_response,
            second_adversarial_response_http_response,
            second_objective_response_http_response,
        ]

        target_response = await red_teaming_orchestrator._retrieve_and_send_prompt_async(
            objective="some objective",
            objective_target_conversation_id=prompt_target_conversation_id,
            adversarial_chat_conversation_id=red_teaming_chat_conversation_id,
        )
        assert target_response.converted_value == "First objective response"

        # The call to objective target is the second http call
        # This verifies the text sent to objective target came from the adversarial chat
        first_objective_target_call = mock_make_request.call_args_list[1]
        first_objective_text = first_objective_target_call[1]["request_body"]["messages"][0]["content"]
        assert first_objective_text == "First adversarial response"

        _check_orchestrator_memory(memory=red_teaming_orchestrator._memory, num_turns=1)

        target_response = await red_teaming_orchestrator._retrieve_and_send_prompt_async(
            objective="some objective",
            objective_target_conversation_id=prompt_target_conversation_id,
            adversarial_chat_conversation_id=red_teaming_chat_conversation_id,
        )
        assert target_response.converted_value == "Second objective response"

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
    with patch.object(CentralMemory, "get_memory_instance", return_value=MagicMock()):
        orchestrator = RedTeamingOrchestrator(
            adversarial_chat=MagicMock(),
            objective_target=MagicMock(),
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
        actual_result = await orchestrator._check_conversation_complete_async(
            objective_target_conversation_id=str(uuid4())
        )
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
    with patch.object(CentralMemory, "get_memory_instance", return_value=MagicMock()):
        orchestrator = RedTeamingOrchestrator(
            adversarial_chat=MagicMock(),
            objective_target=MagicMock(),
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
            await orchestrator._check_conversation_complete_async(objective_target_conversation_id=str(uuid4()))


@pytest.mark.asyncio
@pytest.mark.parametrize("max_turns", [1, 3, 5])
@patch(
    "pyrit.common.default_values.get_non_required_value",
    return_value='{"op_name": "dummy_op", "username": "dummy_user"}',
)
async def test_run_attack_async(
    prompt_target: PromptTarget,
    chat_completion_engine: OpenAIChatTarget,
    red_team_system_prompt_path: pathlib.Path,
    max_turns: int,
):
    scorer = MagicMock(Scorer)
    scorer.scorer_type = "true_false"
    red_teaming_orchestrator = RedTeamingOrchestrator(
        adversarial_chat=chat_completion_engine,
        adversarial_chat_system_prompt_path=red_team_system_prompt_path,
        objective_target=prompt_target,
        max_turns=max_turns,
        objective_scorer=scorer,
    )

    with (
        patch.object(red_teaming_orchestrator, "_retrieve_and_send_prompt_async") as mock_send_prompt,
        patch.object(red_teaming_orchestrator, "_check_conversation_complete_async") as mock_check_complete,
    ):

        mock_send_prompt.return_value = MagicMock(response_error="none")
        mock_check_complete.return_value = MagicMock(get_value=MagicMock(return_value=True))

        result = await red_teaming_orchestrator.run_attack_async(
            objective="objective", memory_labels={"username": "user"}
        )

        assert result is not None
        assert result.conversation_id is not None
        assert result.status == "success"
        assert mock_send_prompt.call_count <= max_turns
        assert mock_check_complete.call_count <= max_turns
        # Test that the global memory labels and passed-in memory labels were combined properly
        assert mock_send_prompt.call_args.kwargs["memory_labels"] == {"op_name": "dummy_op", "username": "user"}


@pytest.mark.asyncio
async def test_run_attack_async_blocked_response(
    prompt_target: PromptTarget,
    chat_completion_engine: OpenAIChatTarget,
    red_team_system_prompt_path: pathlib.Path,
):
    scorer = MagicMock(Scorer)
    scorer.scorer_type = "true_false"
    red_teaming_orchestrator = RedTeamingOrchestrator(
        adversarial_chat=chat_completion_engine,
        adversarial_chat_system_prompt_path=red_team_system_prompt_path,
        objective_target=prompt_target,
        objective_scorer=scorer,
        max_turns=5,
    )

    with patch.object(red_teaming_orchestrator, "_retrieve_and_send_prompt_async") as mock_send_prompt:
        mock_send_prompt.return_value = MagicMock(response_error="blocked")

        result = await red_teaming_orchestrator.run_attack_async(objective="objective")

        assert result.conversation_id is not None
        assert result.status == "failure"
        assert mock_send_prompt.call_count == 5


@pytest.mark.asyncio
async def test_apply_run_attack_async_runtime_error(
    prompt_target: PromptTarget,
    chat_completion_engine: OpenAIChatTarget,
):
    scorer = MagicMock(Scorer)
    scorer.scorer_type = "true_false"
    red_teaming_orchestrator = RedTeamingOrchestrator(
        adversarial_chat=chat_completion_engine,
        objective_target=prompt_target,
        objective_scorer=scorer,
        max_turns=5,
    )

    with patch.object(red_teaming_orchestrator, "_retrieve_and_send_prompt_async") as mock_send_prompt:
        mock_send_prompt.return_value = MagicMock(response_error="unexpected_error")

        with pytest.raises(RuntimeError):
            await red_teaming_orchestrator.run_attack_async(objective="objective")


def test_handle_last_prepended_user_message_with_prepended_message():
    scorer = MagicMock(Scorer)
    scorer.scorer_type = "true_false"
    orchestrator = RedTeamingOrchestrator(
        adversarial_chat=MagicMock(),
        objective_target=MagicMock(),
        objective_scorer=scorer,
    )
    orchestrator._last_prepended_user_message = "Last user message"
    orchestrator._last_prepended_assistant_message_scores = []

    custom_prompt = orchestrator._handle_last_prepended_user_message()
    assert custom_prompt == "Last user message"


def test_handle_last_prepended_assistant_message_with_scores():
    scorer = MagicMock(Scorer)
    scorer.scorer_type = "true_false"
    scorer.get_identifier.return_value = {"__type__": "true_false_scorer"}
    orchestrator = RedTeamingOrchestrator(
        adversarial_chat=MagicMock(),
        objective_target=MagicMock(),
        objective_scorer=scorer,
    )
    score1 = MagicMock(Score)
    score1.scorer_class_identifier = {"__type__": "true_false_scorer"}
    score2 = MagicMock(Score)
    score2.scorer_class_identifier = {"__type__": "other_scorer"}
    orchestrator._last_prepended_assistant_message_scores = [score2, score1]

    objective_score = orchestrator._handle_last_prepended_assistant_message()
    assert objective_score == score1


def test_handle_last_prepended_assistant_message_with_no_matching_score():
    scorer = MagicMock(Scorer)
    scorer.scorer_type = "true_false"
    scorer.get_identifier.return_value = {"__type__": "true_false_scorer"}
    orchestrator = RedTeamingOrchestrator(
        adversarial_chat=MagicMock(),
        objective_target=MagicMock(),
        objective_scorer=scorer,
    )
    score = MagicMock(Score)
    score.scorer_class_identifier = {"__type__": "other_scorer"}
    orchestrator._last_prepended_assistant_message_scores = [score]

    objective_score = orchestrator._handle_last_prepended_assistant_message()
    assert objective_score is None
