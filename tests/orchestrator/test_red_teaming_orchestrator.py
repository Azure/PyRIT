# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import pathlib
import pytest

from typing import Dict, Generator, List, Union
from unittest.mock import Mock, patch

from pyrit.memory.memory_interface import MemoryInterface
from pyrit.prompt_target.prompt_target import PromptTarget
from pyrit.orchestrator import ScoringRedTeamingOrchestrator, EndTokenRedTeamingOrchestrator
from pyrit.orchestrator.end_token_red_teaming_orchestrator import RED_TEAM_CONVERSATION_END_TOKEN
from pyrit.prompt_target import AzureOpenAIChatTarget
from pyrit.models import AttackStrategy, ChatMessage, Score
from pyrit.common.path import DATASETS_PATH

from tests.mocks import get_memory_interface


@pytest.fixture
def memory_interface() -> Generator[MemoryInterface, None, None]:
    yield from get_memory_interface()


@pytest.fixture
def chat_completion_engine() -> AzureOpenAIChatTarget:
    return AzureOpenAIChatTarget(deployment_name="test", endpoint="test", api_key="test")


@pytest.fixture
def prompt_target(memory_interface) -> AzureOpenAIChatTarget:
    return AzureOpenAIChatTarget(
        deployment_name="test",
        endpoint="test",
        api_key="test",
        memory=memory_interface,
    )


@pytest.fixture
def simple_attack_strategy() -> AttackStrategy:
    return AttackStrategy(
        strategy=pathlib.Path(DATASETS_PATH) / "orchestrators" / "red_teaming" / "end_token.yaml",
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


@pytest.mark.parametrize("attack_strategy_as_str", [True, False])
@pytest.mark.parametrize("OrchestratorType", [ScoringRedTeamingOrchestrator, EndTokenRedTeamingOrchestrator])
def test_send_prompt_twice(
    prompt_target: PromptTarget,
    chat_completion_engine: AzureOpenAIChatTarget,
    simple_attack_strategy: AttackStrategy,
    memory_interface: MemoryInterface,
    attack_strategy_as_str: bool,
    OrchestratorType: type,
):
    attack_strategy: Union[str | AttackStrategy] = (
        str(simple_attack_strategy) if attack_strategy_as_str else simple_attack_strategy
    )

    kwargs = {
        "red_teaming_chat": chat_completion_engine,
        "memory": memory_interface,
        "attack_strategy": attack_strategy,
        "initial_red_teaming_prompt": "how can I help you?",
        "prompt_target": prompt_target,
    }
    if OrchestratorType == ScoringRedTeamingOrchestrator:
        kwargs["scorer"] = Mock()
    red_teaming_orchestrator = OrchestratorType(**kwargs)

    with patch.object(red_teaming_orchestrator._red_teaming_chat, "_complete_chat") as mock_rt:
        with patch.object(red_teaming_orchestrator._prompt_target, "_complete_chat") as mock_target:
            mock_rt.return_value = "First red teaming chat response"
            expected_target_response = "First target response"
            mock_target.return_value = expected_target_response
            target_response = red_teaming_orchestrator.send_prompt()
            assert target_response == expected_target_response

            _check_orchestrator_memory_if_no_original_prompt(memory=red_teaming_orchestrator._memory, num_turns=1)

            mock_rt.assert_called_once()
            mock_target.assert_called_once()

            second_target_response = "Second target response"
            mock_rt.return_value = "Second red teaming chat response"
            mock_target.return_value = second_target_response
            target_response = red_teaming_orchestrator.send_prompt()
            assert target_response == second_target_response

            _check_orchestrator_memory_if_no_original_prompt(memory=red_teaming_orchestrator._memory, num_turns=2)


@pytest.mark.parametrize("attack_strategy_as_str", [True, False])
@pytest.mark.parametrize("OrchestratorType", [ScoringRedTeamingOrchestrator, EndTokenRedTeamingOrchestrator])
def test_send_fixed_prompt_then_generated_prompt(
    prompt_target: PromptTarget,
    chat_completion_engine: AzureOpenAIChatTarget,
    simple_attack_strategy: AttackStrategy,
    memory_interface: MemoryInterface,
    attack_strategy_as_str: bool,
    OrchestratorType: type,
):
    attack_strategy: Union[str | AttackStrategy] = (
        str(simple_attack_strategy) if attack_strategy_as_str else simple_attack_strategy
    )

    kwargs = {
        "red_teaming_chat": chat_completion_engine,
        "memory": memory_interface,
        "attack_strategy": attack_strategy,
        "initial_red_teaming_prompt": "how can I help you?",
        "prompt_target": prompt_target,
    }
    if OrchestratorType == ScoringRedTeamingOrchestrator:
        kwargs["scorer"] = Mock()
    red_teaming_orchestrator = OrchestratorType(**kwargs)

    with patch.object(red_teaming_orchestrator._red_teaming_chat, "_complete_chat") as mock_rt:
        with patch.object(red_teaming_orchestrator._prompt_target, "_complete_chat") as mock_target:
            fixed_input_prompt = "First prompt to target - set by user"
            expected_target_responses = "First target response"
            mock_target.return_value = expected_target_responses
            target_response = red_teaming_orchestrator.send_prompt(prompt=fixed_input_prompt)
            assert target_response == expected_target_responses

            _check_orchestrator_memory_if_original_prompt(memory=red_teaming_orchestrator._memory, num_turns=1)

            mock_rt.assert_not_called()
            mock_target.assert_called_once()

            expected_generated_red_teaming_response = "red teaming chat response"
            expected_target_response = "second chat response"

            mock_rt.return_value = expected_generated_red_teaming_response
            mock_target.return_value = expected_target_response

            target_response = red_teaming_orchestrator.send_prompt()
            assert target_response == expected_target_response

            _check_orchestrator_memory_if_original_prompt(memory=red_teaming_orchestrator._memory, num_turns=2)


@pytest.mark.parametrize("attack_strategy_as_str", [True, False])
@pytest.mark.parametrize("OrchestratorType", [ScoringRedTeamingOrchestrator, EndTokenRedTeamingOrchestrator])
def test_send_fixed_prompt_beyond_first_iteration_failure(
    prompt_target: PromptTarget,
    chat_completion_engine: AzureOpenAIChatTarget,
    simple_attack_strategy: AttackStrategy,
    memory_interface: MemoryInterface,
    attack_strategy_as_str: bool,
    OrchestratorType: type,
):
    attack_strategy: Union[str | AttackStrategy] = (
        str(simple_attack_strategy) if attack_strategy_as_str else simple_attack_strategy
    )

    kwargs = {
        "red_teaming_chat": chat_completion_engine,
        "memory": memory_interface,
        "attack_strategy": attack_strategy,
        "initial_red_teaming_prompt": "how can I help you?",
        "prompt_target": prompt_target,
    }
    if OrchestratorType == ScoringRedTeamingOrchestrator:
        kwargs["scorer"] = Mock()
    red_teaming_orchestrator = OrchestratorType(**kwargs)

    with patch.object(red_teaming_orchestrator._red_teaming_chat, "_complete_chat") as mock_rt:
        with patch.object(red_teaming_orchestrator._prompt_target, "_complete_chat") as mock_target:
            fixed_input_prompt = "First prompt to target - set by user"
            expected_target_response = "First target response"
            mock_target.return_value = expected_target_response
            target_response = red_teaming_orchestrator.send_prompt(prompt=fixed_input_prompt)
            assert target_response == expected_target_response
            _check_orchestrator_memory_if_original_prompt(memory=red_teaming_orchestrator._memory, num_turns=1)

            mock_rt.assert_not_called()
            mock_target.assert_called_once()

            second_fixed_input_prompt = "Second prompt to target - sent by user"
            expected_target_response = "Second target response"
            mock_target.return_value = expected_target_response
            with pytest.raises(ValueError):
                target_response = red_teaming_orchestrator.send_prompt(prompt=second_fixed_input_prompt)
            mock_rt.assert_not_called()


@pytest.mark.parametrize("attack_strategy_as_str", [True, False])
def test_reach_goal_after_two_turns_end_token(
    prompt_target: PromptTarget,
    chat_completion_engine: AzureOpenAIChatTarget,
    simple_attack_strategy: AttackStrategy,
    memory_interface: MemoryInterface,
    attack_strategy_as_str: bool,
):
    attack_strategy: Union[str | AttackStrategy] = (
        str(simple_attack_strategy) if attack_strategy_as_str else simple_attack_strategy
    )

    red_teaming_orchestrator = EndTokenRedTeamingOrchestrator(
        red_teaming_chat=chat_completion_engine,
        memory=memory_interface,
        attack_strategy=attack_strategy,
        initial_red_teaming_prompt="how can I help you?",
        prompt_target=prompt_target,
        verbose=True,
    )

    with patch.object(red_teaming_orchestrator._red_teaming_chat, "_complete_chat") as mock_rt:
        with patch.object(red_teaming_orchestrator._prompt_target, "_complete_chat") as mock_target:
            expected_red_teaming_responses = [
                "First red teaming chat response",
                "Second red teaming chat response with expected end token <|done|>",
            ]
            mock_rt.side_effect = expected_red_teaming_responses
            expected_target_response = "First target response"
            mock_target.return_value = expected_target_response
            target_response = red_teaming_orchestrator.apply_attack_strategy_until_completion()
            assert target_response == expected_target_response

            _check_orchestrator_memory_if_original_prompt(memory=red_teaming_orchestrator._memory, num_turns=2)

            assert mock_rt.call_count == 2
            mock_target.assert_called_once()


def test_default_attack_strategy_set_with_end_token():
    valid_strategy = "This is a strategy! Print <|done|> at the end."
    orchestrator = EndTokenRedTeamingOrchestrator(
        attack_strategy=valid_strategy,
        red_teaming_chat=Mock(),
        prompt_target=Mock(),
        memory=Mock(),
    )
    assert orchestrator._attack_strategy is not None
    assert RED_TEAM_CONVERSATION_END_TOKEN in orchestrator._attack_strategy


def test_attack_strategy_without_token_raises_with_end_token_orchestrator():
    tokenless_strategy = "This is a strategy without end token."
    with pytest.raises(ValueError):
        EndTokenRedTeamingOrchestrator(
            attack_strategy=tokenless_strategy,
            red_teaming_chat=Mock(),
            prompt_target=Mock(),
            memory=Mock(),
        )


def test_attack_strategy_without_token_does_not_raise_with_scoring_orchestrator():
    tokenless_strategy = "This is a strategy without end token."
    ScoringRedTeamingOrchestrator(
        attack_strategy=tokenless_strategy,
        red_teaming_chat=Mock(),
        prompt_target=Mock(),
        memory=Mock(),
        scorer=Mock(),
    )


@pytest.mark.parametrize("score", [True, False])
@pytest.mark.parametrize("message_count", [0, 1, 2, 3, 4, 5, 9, 10, 11, 99, 100, 101])
@pytest.mark.parametrize("red_teaming_chat_role", ["user", "assistant"])
def test_is_conversation_complete_scoring(score, message_count, red_teaming_chat_role):
    scorer = Mock()
    scorer.score_text = Mock(return_value=Score(score_type="bool", score_value=score))
    orchestrator = ScoringRedTeamingOrchestrator(
        attack_strategy="some strategy",
        red_teaming_chat=Mock(),
        prompt_target=Mock(),
        memory=Mock(),
        scorer=scorer,
    )
    # simulate back and forth between user and assistant
    messages = [
        ChatMessage(role="user" if i % 2 == 0 else "assistant", content=f"Message #{i}")
        for i in range(message_count - 1)
    ]
    # conversation is complete if the last message is from the target
    # and the score is True
    assert orchestrator.is_conversation_complete(messages, red_teaming_chat_role=red_teaming_chat_role) == (
        len(messages) > 0 and red_teaming_chat_role != messages[-1].role and score
    )


@pytest.mark.parametrize(
    "score",
    [
        Score(score_type="float", score_value=1.7),
        Score(score_type="str", score_value="abc"),
        Score(score_type="int", score_value=89),
    ],
)
def test_is_conversation_complete_scoring_non_bool(score):
    scorer = Mock()
    scorer.score_text = Mock(return_value=score)
    orchestrator = ScoringRedTeamingOrchestrator(
        attack_strategy="some strategy",
        red_teaming_chat=Mock(),
        prompt_target=Mock(),
        memory=Mock(),
        scorer=scorer,
    )
    with pytest.raises(ValueError):
        orchestrator.is_conversation_complete(
            messages=[
                ChatMessage(role="user", content="First message."),
                ChatMessage(role="assistant", content="Second message."),
                ChatMessage(role="user", content="Third message."),
                ChatMessage(role="assistant", content="Fourth message."),
            ],
            red_teaming_chat_role="user",
        )


@pytest.fixture
def end_token_red_teaming_orchestrator():
    valid_strategy = "This is a strategy! Print <|done|> at the end."
    return EndTokenRedTeamingOrchestrator(
        attack_strategy=valid_strategy,
        red_teaming_chat=Mock(),
        prompt_target=Mock(),
        memory=Mock(),
    )


@pytest.mark.parametrize("message_count", [0, 1, 2, 3, 4, 5, 9, 10, 11, 99, 100, 101])
@pytest.mark.parametrize("end_token_present", [True, False])
@pytest.mark.parametrize("red_teaming_chat_role", ["user", "assistant"])
def test_is_conversation_complete_end_token(
    end_token_red_teaming_orchestrator, message_count, end_token_present, red_teaming_chat_role
):
    # simulate back and forth between user and assistant
    messages = [
        ChatMessage(role="user" if i % 2 == 0 else "assistant", content=f"Message #{i}")
        for i in range(message_count - 1)
    ]
    last_message_role = "user" if message_count - 1 % 2 == 0 else "assistant"
    last_message_content = f"Message #{message_count - 1}"
    if end_token_present:
        last_message_content += " <|done|>"
    if message_count > 0:
        messages.append(ChatMessage(role=last_message_role, content=last_message_content))
        # if the red teaming endpoint is the assistant we can also set the system prompt
        messages = [ChatMessage(role="system", content="The system prompt.")] + messages
    # the conversation is complete if the red teaming bot as indicated by
    # red_teaming_chat_role sends the end token
    assert end_token_red_teaming_orchestrator.is_conversation_complete(
        messages, red_teaming_chat_role=red_teaming_chat_role
    ) == (end_token_present and last_message_role == red_teaming_chat_role and message_count > 0)
