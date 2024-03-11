# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import pathlib
from unittest.mock import Mock, patch
from pyrit.prompt_target.prompt_target import PromptTarget

import pytest
from openai.types.chat import ChatCompletion, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice

from pyrit.orchestrator import ScoringRedTeamingOrchestrator, EndTokenRedTeamingOrchestrator
from pyrit.orchestrator.end_token_red_teaming_orchestrator import RED_TEAM_CONVERSATION_END_TOKEN
from pyrit.chat import AzureOpenAIChat
from pyrit.prompt_target import AzureOpenAIChatTarget
from pyrit.models import AttackStrategy
from pyrit.memory import FileMemory
from pyrit.common.path import DATASETS_PATH


@pytest.fixture
def openai_mock_return() -> ChatCompletion:
    return ChatCompletion(
        id="12345678-1a2b-3c4e5f-a123-12345678abcd",
        object="chat.completion",
        choices=[
            Choice(
                index=0,
                message=ChatCompletionMessage(role="assistant", content="hi, I'm adversary chat."),
                finish_reason="stop",
                logprobs=None,
            )
        ],
        created=1629389505,
        model="gpt-4",
    )


@pytest.fixture
def memory(tmp_path: pathlib.Path) -> FileMemory:
    return FileMemory(filepath=tmp_path / "test.json.memory")


@pytest.fixture
def chat_completion_engine() -> AzureOpenAIChat:
    return AzureOpenAIChat(deployment_name="test", endpoint="test", api_key="test")


@pytest.fixture
def prompt_target(memory) -> AzureOpenAIChatTarget:
    return AzureOpenAIChatTarget(
        deployment_name="test",
        endpoint="test",
        api_key="test",
        memory=memory,
    )

@pytest.fixture
def simple_attack_strategy() -> AttackStrategy:
    return AttackStrategy(
        strategy=pathlib.Path(DATASETS_PATH) / "attack_strategies" / "multi_turn_chat" / "red_team_chatbot_with_objective.yaml",
        conversation_objective="Do bad stuff",
    )


def check_conversations(conversations, red_teaming_meta_prompt, initial_red_teaming_prompt, expected_red_teaming_responses, expected_target_responses, stop_after_n_conversations):
    # first conversation (with red teaming chat bot)
    assert conversations[0].conversation_id == conversations[1].conversation_id == conversations[2].conversation_id
    assert conversations[0].role == "system"
    assert conversations[0].content == red_teaming_meta_prompt
    assert conversations[1].role == "user"
    assert conversations[1].content == initial_red_teaming_prompt
    assert conversations[2].role == "assistant"
    assert conversations[2].content == expected_red_teaming_responses[0]
    # second conversation (with prompt target)
    assert conversations[3].conversation_id == conversations[4].conversation_id
    assert conversations[3].normalizer_id == conversations[4].normalizer_id
    assert conversations[3].role == "user"
    assert conversations[3].content == expected_red_teaming_responses[0]
    assert conversations[4].role == "assistant"
    assert conversations[4].content == expected_target_responses[0]

    if stop_after_n_conversations == 2:
        return
    
    # third conversation (with red teaming chatbot)
    assert conversations[5].conversation_id == conversations[6].conversation_id
    assert conversations[5].role == "user"
    assert conversations[5].content == expected_target_responses[0]
    assert conversations[6].role == "assistant"
    assert conversations[6].content == expected_red_teaming_responses[1]

    if stop_after_n_conversations == 3:
        return

    # fourth conversation (with prompt target)
    assert conversations[7].conversation_id == conversations[8].conversation_id
    assert conversations[7].normalizer_id == conversations[8].normalizer_id
    assert conversations[7].role == "user"
    assert conversations[7].content == expected_red_teaming_responses[1]
    assert conversations[8].role == "assistant"
    assert conversations[8].content == expected_target_responses[1]


@pytest.mark.parametrize(
    "attack_strategy_as_str", [True, False]
)
@pytest.mark.parametrize(
    "OrchestratorType", [ScoringRedTeamingOrchestrator, EndTokenRedTeamingOrchestrator]
)
def test_send_prompt_twice(
        prompt_target: PromptTarget,
        chat_completion_engine: AzureOpenAIChat,
        simple_attack_strategy: AttackStrategy,
        memory: FileMemory,
        attack_strategy_as_str: bool,
        OrchestratorType: type):
    attack_strategy = simple_attack_strategy
    if attack_strategy_as_str:
        attack_strategy = str(simple_attack_strategy)

    kwargs = {
        "red_teaming_chat": chat_completion_engine,
        "memory": memory,
        "attack_strategy": attack_strategy,
        "initial_red_teaming_prompt": "how can I help you?",
        "prompt_target": prompt_target,
    }
    if OrchestratorType == ScoringRedTeamingOrchestrator:
        kwargs["scorer"] = Mock()
    red_teaming_orchestrator = OrchestratorType(**kwargs)

    with patch.object(red_teaming_orchestrator._red_teaming_chat, "complete_chat") as mock_rt:
        with patch.object(red_teaming_orchestrator._prompt_target, "complete_chat") as mock_target:
            expected_red_teaming_responses = ["First red teaming chat response"]
            mock_rt.return_value = expected_red_teaming_responses[0]
            expected_target_responses = ["First target response"]
            mock_target.return_value = expected_target_responses[0]
            target_response = red_teaming_orchestrator.send_prompt()
            assert target_response == expected_target_responses
            conversations = red_teaming_orchestrator._memory.get_all_memory()
            # Expecting two conversation threads (one with red teaming chat and one with prompt target)
            assert len(conversations) == 5, f"Expected 5 conversations, got {len(conversations)}"
            check_conversations(conversations, str(attack_strategy), red_teaming_orchestrator._initial_red_teaming_prompt, expected_red_teaming_responses, expected_target_responses, stop_after_n_conversations=2)
            
            mock_rt.assert_called_once()
            mock_target.assert_called_once()

            expected_red_teaming_responses.append("Second red teaming chat response")
            mock_rt.return_value = expected_red_teaming_responses[1]
            expected_target_responses.append("Second target response")
            mock_target.return_value = expected_target_responses[1]
            target_response = red_teaming_orchestrator.send_prompt()
            assert target_response == [expected_target_responses[1]]
            conversations = red_teaming_orchestrator._memory.get_all_memory()
            # Expecting another two conversation threads
            assert len(conversations) == 9, f"Expected 9 conversations, got {len(conversations)}"
            check_conversations(conversations, str(attack_strategy), red_teaming_orchestrator._initial_red_teaming_prompt, expected_red_teaming_responses, expected_target_responses, stop_after_n_conversations=4)


@pytest.mark.parametrize(
    "attack_strategy_as_str", [True, False]
)
def test_reach_goal_after_two_turns_end_token(
        prompt_target: PromptTarget,
        chat_completion_engine: AzureOpenAIChat,
        simple_attack_strategy: AttackStrategy,
        memory: FileMemory,
        attack_strategy_as_str: bool):
    attack_strategy = simple_attack_strategy
    if attack_strategy_as_str:
        attack_strategy = str(simple_attack_strategy)

    red_teaming_orchestrator = EndTokenRedTeamingOrchestrator(
        red_teaming_chat=chat_completion_engine,
        memory=memory,
        attack_strategy=attack_strategy,
        initial_red_teaming_prompt="how can I help you?",
        prompt_target=prompt_target,
        verbose=True
    )

    with patch.object(red_teaming_orchestrator._red_teaming_chat, "complete_chat") as mock_rt:
        with patch.object(red_teaming_orchestrator._prompt_target, "complete_chat") as mock_target:
            expected_red_teaming_responses = ["First red teaming chat response", "Second red teaming chat response with expected end token <|done|>"]
            mock_rt.side_effect = expected_red_teaming_responses
            expected_target_response = "First target response"
            mock_target.return_value = expected_target_response
            target_response = red_teaming_orchestrator.apply_attack_strategy_until_completion()
            assert target_response == expected_target_response
            conversations = red_teaming_orchestrator._memory.get_all_memory()
            # Expecting three conversation threads (two with red teaming chat and one with prompt target)
            assert len(conversations) == 7, f"Expected 7 conversations, got {len(conversations)}"
            check_conversations(conversations, str(attack_strategy), red_teaming_orchestrator._initial_red_teaming_prompt, expected_red_teaming_responses, [expected_target_response], stop_after_n_conversations=3)
            
            assert mock_rt.call_count == 2
            mock_target.assert_called_once()


def test_is_conversation_complete_false(red_teaming_bot):
    with patch.object(red_teaming_bot._chat_engine, "complete_chat") as mock:
        mock.return_value = "Hello, this is a message sent by the assistant. How can i help you?"
        red_teaming_bot.complete_chat_user("hi, I am a victim chatbot, how can I help?")
        red_teaming_bot.complete_chat_user("hi, I am a victim chatbot, how can I help?")

        assert (
            red_teaming_bot.is_conversation_complete() is False
        ), "Conversation should not be complete, objective not realized"


def test_is_conversation_complete_emptyhistory_false(red_teaming_bot):
    with patch.object(red_teaming_bot._chat_engine, "complete_chat") as mock:
        mock.return_value = "Hello, this is a message sent by the assistant. How can i help you?"
        assert red_teaming_bot.is_conversation_complete() is False, "Conversation is empty, objective not realized"


def test_is_conversation_complete_true(red_teaming_bot):
    with patch.object(red_teaming_bot._chat_engine, "complete_chat") as mock:
        mock.return_value = "end token <|done|>"
        red_teaming_bot.complete_chat_user(message="bad stuff is done")
        assert (
            red_teaming_bot.is_conversation_complete() is True
        ), "Conversation should be complete, objective is realized"


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


def test_attack_strategy_without_token_raises():
    invalid_strategy = "This is a bad strategy"
    with pytest.raises(ValueError):
        EndTokenRedTeamingOrchestrator(
            attack_strategy=invalid_strategy,
            red_teaming_chat=Mock(),
            prompt_target=Mock(),
            memory=Mock(),
        )
