# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import pathlib
from typing import Dict, List
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from pyrit.common.path import DATASETS_PATH
from pyrit.memory import CentralMemory
from pyrit.models import PromptRequestPiece, PromptRequestResponse, Score
from pyrit.orchestrator import RedTeamingOrchestrator
from pyrit.prompt_target import OpenAIChatTarget, PromptTarget
from pyrit.score import Scorer


@pytest.fixture
def chat_completion_engine(patch_central_database) -> OpenAIChatTarget:
    return OpenAIChatTarget(deployment_name="test", endpoint="test", api_key="test")


@pytest.fixture
def prompt_target(patch_central_database) -> OpenAIChatTarget:
    return OpenAIChatTarget(deployment_name="test", endpoint="test", api_key="test")


@pytest.fixture
def red_team_system_prompt_path() -> pathlib.Path:
    return pathlib.Path(DATASETS_PATH) / "orchestrators" / "red_teaming" / "text_generation.yaml"


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


@pytest.mark.asyncio
async def test_send_prompt_twice(
    prompt_target: PromptTarget,
    chat_completion_engine: OpenAIChatTarget,
):

    scorer = MagicMock(Scorer)
    scorer.scorer_type = "true_false"
    red_teaming_orchestrator = RedTeamingOrchestrator(
        adversarial_chat=chat_completion_engine,
        objective_target=prompt_target,
        objective_scorer=scorer,
    )

    prompt_target_conversation_id = str(uuid4())
    red_teaming_chat_conversation_id = str(uuid4())
    with patch.object(red_teaming_orchestrator._adversarial_chat, "_complete_chat_async") as mock_rt:
        with patch.object(red_teaming_orchestrator._objective_target, "_complete_chat_async") as mock_target:
            mock_rt.return_value = "First red teaming chat response"
            expected_target_response = "First target response"
            mock_target.return_value = expected_target_response
            target_response = await red_teaming_orchestrator._retrieve_and_send_prompt_async(
                objective="some objective",
                objective_target_conversation_id=prompt_target_conversation_id,
                adversarial_chat_conversation_id=red_teaming_chat_conversation_id,
            )
            assert target_response.converted_value == expected_target_response

            _check_orchestrator_memory(memory=red_teaming_orchestrator._memory, num_turns=1)

            mock_rt.assert_called_once()
            mock_target.assert_called_once()

            second_target_response = "Second target response"
            mock_rt.return_value = "Second red teaming chat response"
            mock_target.return_value = second_target_response
            target_response = await red_teaming_orchestrator._retrieve_and_send_prompt_async(
                objective="some objective",
                objective_target_conversation_id=prompt_target_conversation_id,
                adversarial_chat_conversation_id=red_teaming_chat_conversation_id,
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
        assert result.achieved_objective is True
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
        assert red_teaming_orchestrator._achieved_objective is False
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

@pytest.mark.asyncio
async def test_get_prompt_from_adversarial_chat_does_not_call_system_prompt(patch_central_database):
    scorer = MagicMock(Scorer)
    scorer.scorer_type = "true_false"
    orchestrator = RedTeamingOrchestrator(
        adversarial_chat=MagicMock(),
        objective_target=MagicMock(),
        objective_scorer=scorer,
    )

    prompt_request_response = PromptRequestResponse(request_pieces=[
        PromptRequestPiece(role="user", original_value="message", converted_value="message",
                           original_value_data_type="text", converted_value_data_type="text", )])
    conversations = [
        prompt_request_response
    ]

    with (
        patch.object(
            orchestrator, "_get_prompt_for_adversarial_chat", MagicMock(return_value="prompt_text")
        ),
        patch.object(
            orchestrator._memory, "get_conversation", MagicMock(return_value=conversations)
        ),
        patch.object(
            orchestrator._prompt_normalizer, "send_prompt_async", AsyncMock(return_value=prompt_request_response)
        ),
        patch.object(
            orchestrator._adversarial_chat, "set_system_prompt", AsyncMock()
        ) as mock_set_system_prompt,
    ):
        objective = "objective"
        memory_labels = {"username": "user"}
        objective_target_conversation_id = str(uuid4())
        adversarial_chat_conversation_id = str(uuid4())

        await orchestrator._get_prompt_from_adversarial_chat(
            objective=objective,
            objective_target_conversation_id=objective_target_conversation_id,
            adversarial_chat_conversation_id=adversarial_chat_conversation_id,
            memory_labels=memory_labels
        )

        mock_set_system_prompt.assert_not_called()

@pytest.mark.asyncio
async def test_get_prompt_from_adversarial_chat_does_call_system_prompt(patch_central_database):
    scorer = MagicMock(Scorer)
    scorer.scorer_type = "true_false"
    orchestrator = RedTeamingOrchestrator(
        adversarial_chat=MagicMock(),
        objective_target=MagicMock(),
        objective_scorer=scorer,
    )

    prompt_request_response = PromptRequestResponse(request_pieces=[
        PromptRequestPiece(role="user", original_value="message", converted_value="message",
                           original_value_data_type="text", converted_value_data_type="text", )])

    with (
        patch.object(
            orchestrator, "_get_prompt_for_adversarial_chat", MagicMock(return_value="prompt_text")
        ),
        patch.object(
            orchestrator._memory, "get_conversation", MagicMock(return_value=[])
        ),
        patch.object(
            orchestrator._prompt_normalizer, "send_prompt_async", AsyncMock(return_value=prompt_request_response)
        ),
        patch.object(
            orchestrator._adversarial_chat, "set_system_prompt", AsyncMock()
        ) as mock_set_system_prompt,
    ):
        objective = "objective"
        memory_labels = {"username": "user"}
        objective_target_conversation_id = str(uuid4())
        adversarial_chat_conversation_id = str(uuid4())

        await orchestrator._get_prompt_from_adversarial_chat(
            objective=objective,
            objective_target_conversation_id=objective_target_conversation_id,
            adversarial_chat_conversation_id=adversarial_chat_conversation_id,
            memory_labels=memory_labels
        )

        expected_system_prompt = orchestrator._adversarial_chat_system_seed_prompt.render_template_value(
            objective=objective,
            max_turns=orchestrator._max_turns,
        )

        mock_set_system_prompt.assert_called_once()

        _, adv_chat_system_prompt_args = mock_set_system_prompt.call_args
        assert adv_chat_system_prompt_args["system_prompt"] == expected_system_prompt
        assert adv_chat_system_prompt_args["conversation_id"] == adversarial_chat_conversation_id
        assert adv_chat_system_prompt_args["orchestrator_identifier"] == orchestrator.get_identifier_with_objective(objective=objective)
        assert adv_chat_system_prompt_args["labels"].items() == memory_labels.items()

@pytest.mark.asyncio
async def test_get_prompt_from_adversarial_chat_sends_prompt(patch_central_database):
    scorer = MagicMock(Scorer)
    scorer.scorer_type = "true_false"
    orchestrator = RedTeamingOrchestrator(
        adversarial_chat=MagicMock(),
        objective_target=MagicMock(),
        objective_scorer=scorer,
    )

    prompt_request_response = PromptRequestResponse(request_pieces=[
        PromptRequestPiece(role="user", original_value="message", converted_value="message",
                           original_value_data_type="text", converted_value_data_type="text", )])

    with (
        patch.object(
            orchestrator, "_get_prompt_for_adversarial_chat", MagicMock(return_value="prompt_text")
        ),
        patch.object(
            orchestrator._memory, "get_conversation", MagicMock(return_value=[])
        ),
        patch.object(
            orchestrator._prompt_normalizer, "send_prompt_async", AsyncMock(return_value=prompt_request_response)
        ) as mock_send_prompt,
        patch.object(
            orchestrator._adversarial_chat, "set_system_prompt", AsyncMock()
        ),
    ):
        objective = "objective"
        memory_labels = {"username": "user"}
        objective_target_conversation_id = str(uuid4())
        adversarial_chat_conversation_id = str(uuid4())

        await orchestrator._get_prompt_from_adversarial_chat(
            objective=objective,
            objective_target_conversation_id=objective_target_conversation_id,
            adversarial_chat_conversation_id=adversarial_chat_conversation_id,
            memory_labels=memory_labels
        )

        mock_send_prompt.assert_called_once()

        _, adv_target_send_prompt_args = mock_send_prompt.call_args
        assert len(adv_target_send_prompt_args["seed_prompt_group"].prompts) == 1
        assert adv_target_send_prompt_args["seed_prompt_group"].prompts[0].value == "prompt_text"
        assert adv_target_send_prompt_args["conversation_id"] == adversarial_chat_conversation_id
        assert adv_target_send_prompt_args["target"] == orchestrator._adversarial_chat
        assert adv_target_send_prompt_args["orchestrator_identifier"] == orchestrator.get_identifier_with_objective(objective=objective)
        assert adv_target_send_prompt_args["labels"].items() == memory_labels.items()


@pytest.mark.asyncio
async def test_retrieve_and_sends_prompt(patch_central_database):
    scorer = MagicMock(Scorer)
    scorer.scorer_type = "true_false"
    orchestrator = RedTeamingOrchestrator(
        adversarial_chat=MagicMock(),
        objective_target=MagicMock(),
        objective_scorer=scorer,
    )

    prompt_request_response = PromptRequestResponse(request_pieces=[
        PromptRequestPiece(role="user", original_value="message", converted_value="message",
                           original_value_data_type="text", converted_value_data_type="text", )])

    with (
        patch.object(
            orchestrator, "_get_prompt_from_adversarial_chat", AsyncMock(return_value="prompt_text")
        ),
        patch.object(
            orchestrator._prompt_normalizer, "send_prompt_async", AsyncMock(return_value=prompt_request_response)
        ) as mock_send_prompt,
    ):
        objective = "objective"
        memory_labels = {"username": "user"}
        objective_target_conversation_id = str(uuid4())
        adversarial_chat_conversation_id = str(uuid4())

        await orchestrator._retrieve_and_send_prompt_async(
            objective=objective,
            objective_target_conversation_id=objective_target_conversation_id,
            adversarial_chat_conversation_id=adversarial_chat_conversation_id,
            memory_labels=memory_labels
        )

        mock_send_prompt.assert_called_once()

        _, target_send_prompt_args = mock_send_prompt.call_args
        assert len(target_send_prompt_args["seed_prompt_group"].prompts) == 1
        assert target_send_prompt_args["seed_prompt_group"].prompts[0].value == "prompt_text"
        assert target_send_prompt_args["conversation_id"] == objective_target_conversation_id
        assert target_send_prompt_args["target"] == orchestrator._objective_target
        assert target_send_prompt_args["orchestrator_identifier"] == orchestrator.get_identifier_with_objective(objective=objective)
        assert target_send_prompt_args["labels"].items() == memory_labels.items()
        assert len(target_send_prompt_args["request_converter_configurations"]) == 1
        assert target_send_prompt_args["request_converter_configurations"][0].converters == orchestrator._prompt_converters