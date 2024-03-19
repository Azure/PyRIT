# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import pathlib
from typing import Union
from unittest.mock import Mock, patch
from pyrit.prompt_converter.prompt_converter import PromptConverter
from pyrit.prompt_target.prompt_target import PromptTarget

import pytest
from sqlalchemy import inspect

from pyrit.memory.memory_models import ConversationData
from pyrit.orchestrator import ScoringRedTeamingOrchestrator, EndTokenRedTeamingOrchestrator
from pyrit.orchestrator.end_token_red_teaming_orchestrator import RED_TEAM_CONVERSATION_END_TOKEN
from pyrit.chat import AzureOpenAIChat
from pyrit.prompt_target import AzureOpenAIChatTarget
from pyrit.models import AttackStrategy, ChatMessage, Score
from pyrit.memory import DuckDBMemory
from pyrit.common.path import DATASETS_PATH


@pytest.fixture
def memory() -> DuckDBMemory:  # type: ignore
    # Create an in-memory DuckDB engine
    duckdb_memory = DuckDBMemory(db_path=":memory:")

    # Reset the database to ensure a clean state
    duckdb_memory.reset_database()
    inspector = inspect(duckdb_memory.engine)

    # Verify that tables are created as expected
    assert "ConversationStore" in inspector.get_table_names(), "ConversationStore table not created."
    assert "EmbeddingStore" in inspector.get_table_names(), "EmbeddingStore table not created."

    yield duckdb_memory
    duckdb_memory.dispose_engine()


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
        strategy=pathlib.Path(DATASETS_PATH)
        / "attack_strategies"
        / "multi_turn_chat"
        / "red_team_chatbot_with_objective.yaml",
        conversation_objective="Do bad stuff",
    )


def check_conversations(
    conversations,
    *,
    red_teaming_meta_prompt,
    initial_red_teaming_prompt,
    expected_red_teaming_responses,
    expected_target_responses,
    stop_after_n_conversations=4,
    skip_initial_red_teaming_prompt_generation=False,
):
    index_offset = 0
    if skip_initial_red_teaming_prompt_generation:
        index_offset = 3
    else:
        # first conversation (with red teaming chat bot)
        assert conversations[0].conversation_id == conversations[1].conversation_id == conversations[2].conversation_id
        assert conversations[0].role == "system"
        assert conversations[0].content == red_teaming_meta_prompt
        assert conversations[1].role == "user"
        assert conversations[1].content == initial_red_teaming_prompt
        assert conversations[2].role == "assistant"
        assert conversations[2].content == expected_red_teaming_responses[0]
    # second conversation (with prompt target)
    assert conversations[3 - index_offset].conversation_id == conversations[4 - index_offset].conversation_id
    assert conversations[3 - index_offset].normalizer_id == conversations[4 - index_offset].normalizer_id
    assert conversations[3 - index_offset].role == "user"
    assert conversations[3 - index_offset].content == expected_red_teaming_responses[0]
    assert conversations[4 - index_offset].role == "assistant"
    assert conversations[4 - index_offset].content == expected_target_responses[0]

    if stop_after_n_conversations == 2:
        return

    if skip_initial_red_teaming_prompt_generation:
        assert conversations[2].conversation_id == conversations[3].conversation_id == conversations[4].conversation_id
        assert conversations[2].role == "system"
        index_offset = 2

    # third conversation (with red teaming chatbot)
    assert conversations[5 - index_offset].conversation_id == conversations[6 - index_offset].conversation_id
    assert conversations[5 - index_offset].role == "user"
    assert conversations[5 - index_offset].content == expected_target_responses[0]
    assert conversations[6 - index_offset].role == "assistant"
    assert conversations[6 - index_offset].content == expected_red_teaming_responses[1]

    if stop_after_n_conversations == 3:
        return

    # fourth conversation (with prompt target)
    assert conversations[7 - index_offset].conversation_id == conversations[8 - index_offset].conversation_id
    assert conversations[7 - index_offset].normalizer_id == conversations[8 - index_offset].normalizer_id
    assert conversations[7 - index_offset].role == "user"
    assert conversations[7 - index_offset].content == expected_red_teaming_responses[1]
    assert conversations[8 - index_offset].role == "assistant"
    assert conversations[8 - index_offset].content == expected_target_responses[1]


@pytest.mark.parametrize("attack_strategy_as_str", [True, False])
@pytest.mark.parametrize("OrchestratorType", [ScoringRedTeamingOrchestrator, EndTokenRedTeamingOrchestrator])
def test_send_prompt_twice(
    prompt_target: PromptTarget,
    chat_completion_engine: AzureOpenAIChat,
    simple_attack_strategy: AttackStrategy,
    memory: DuckDBMemory,
    attack_strategy_as_str: bool,
    OrchestratorType: type,
):
    attack_strategy: Union[str | AttackStrategy] = (
        str(simple_attack_strategy) if attack_strategy_as_str else simple_attack_strategy
    )

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
            assert target_response == expected_target_responses[0]
            conversations = red_teaming_orchestrator._memory.get_all_memory(ConversationData)
            # Expecting two conversation threads (one with red teaming chat and one with prompt target)
            assert len(conversations) == 5, f"Expected 5 conversations, got {len(conversations)}"
            check_conversations(
                conversations,
                red_teaming_meta_prompt=str(attack_strategy),
                initial_red_teaming_prompt=red_teaming_orchestrator._initial_red_teaming_prompt,
                expected_red_teaming_responses=expected_red_teaming_responses,
                expected_target_responses=expected_target_responses,
                stop_after_n_conversations=2,
            )

            mock_rt.assert_called_once()
            mock_target.assert_called_once()

            expected_red_teaming_responses.append("Second red teaming chat response")
            mock_rt.return_value = expected_red_teaming_responses[1]
            expected_target_responses.append("Second target response")
            mock_target.return_value = expected_target_responses[1]
            target_response = red_teaming_orchestrator.send_prompt()
            assert target_response == expected_target_responses[1]
            conversations = red_teaming_orchestrator._memory.get_all_memory(ConversationData)
            # Expecting another two conversation threads
            assert len(conversations) == 9, f"Expected 9 conversations, got {len(conversations)}"
            check_conversations(
                conversations,
                red_teaming_meta_prompt=str(attack_strategy),
                initial_red_teaming_prompt=red_teaming_orchestrator._initial_red_teaming_prompt,
                expected_red_teaming_responses=expected_red_teaming_responses,
                expected_target_responses=expected_target_responses,
                stop_after_n_conversations=4,
            )


@pytest.mark.parametrize("attack_strategy_as_str", [True, False])
@pytest.mark.parametrize("OrchestratorType", [ScoringRedTeamingOrchestrator, EndTokenRedTeamingOrchestrator])
def test_send_fixed_prompt_then_generated_prompt(
    prompt_target: PromptTarget,
    chat_completion_engine: AzureOpenAIChat,
    simple_attack_strategy: AttackStrategy,
    memory: DuckDBMemory,
    attack_strategy_as_str: bool,
    OrchestratorType: type,
):
    attack_strategy: Union[str | AttackStrategy] = (
        str(simple_attack_strategy) if attack_strategy_as_str else simple_attack_strategy
    )

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
            fixed_input_prompt = "First prompt to target - set by user"
            expected_target_responses = ["First target response"]
            mock_target.return_value = expected_target_responses[0]
            target_response = red_teaming_orchestrator.send_prompt(prompt=fixed_input_prompt)
            assert target_response == expected_target_responses[0]
            conversations = red_teaming_orchestrator._memory.get_all_memory(ConversationData)
            # Expecting two conversation threads (one with red teaming chat and one with prompt target)
            assert len(conversations) == 2, f"Expected 2 conversations, got {len(conversations)}"
            check_conversations(
                conversations,
                red_teaming_meta_prompt=str(attack_strategy),
                initial_red_teaming_prompt=red_teaming_orchestrator._initial_red_teaming_prompt,
                expected_red_teaming_responses=[fixed_input_prompt],
                expected_target_responses=expected_target_responses,
                stop_after_n_conversations=2,
                skip_initial_red_teaming_prompt_generation=True,
            )

            mock_rt.assert_not_called()
            mock_target.assert_called_once()

            expected_generated_red_teaming_response = "Second red teaming chat response"
            mock_rt.return_value = expected_generated_red_teaming_response
            expected_target_responses.append("Second target response")
            mock_target.return_value = expected_target_responses[1]
            target_response = red_teaming_orchestrator.send_prompt()
            assert target_response == expected_target_responses[1]
            conversations = red_teaming_orchestrator._memory.get_all_memory(ConversationData)
            # Expecting another two conversation threads
            assert len(conversations) == 7, f"Expected 7 conversations, got {len(conversations)}"
            check_conversations(
                conversations,
                red_teaming_meta_prompt=str(attack_strategy),
                initial_red_teaming_prompt=red_teaming_orchestrator._initial_red_teaming_prompt,
                expected_red_teaming_responses=[fixed_input_prompt, expected_generated_red_teaming_response],
                expected_target_responses=expected_target_responses,
                skip_initial_red_teaming_prompt_generation=True,
            )


@pytest.mark.parametrize("attack_strategy_as_str", [True, False])
@pytest.mark.parametrize("OrchestratorType", [ScoringRedTeamingOrchestrator, EndTokenRedTeamingOrchestrator])
def test_send_fixed_prompt_beyond_first_iteration_failure(
    prompt_target: PromptTarget,
    chat_completion_engine: AzureOpenAIChat,
    simple_attack_strategy: AttackStrategy,
    memory: DuckDBMemory,
    attack_strategy_as_str: bool,
    OrchestratorType: type,
):
    attack_strategy: Union[str | AttackStrategy] = (
        str(simple_attack_strategy) if attack_strategy_as_str else simple_attack_strategy
    )

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
            fixed_input_prompt = "First prompt to target - set by user"
            expected_target_responses = ["First target response"]
            mock_target.return_value = expected_target_responses[0]
            target_response = red_teaming_orchestrator.send_prompt(prompt=fixed_input_prompt)
            assert target_response == expected_target_responses[0]
            conversations = red_teaming_orchestrator._memory.get_all_memory(ConversationData)
            # Expecting two conversation threads (one with red teaming chat and one with prompt target)
            assert len(conversations) == 2, f"Expected 2 conversations, got {len(conversations)}"
            check_conversations(
                conversations,
                red_teaming_meta_prompt=str(attack_strategy),
                initial_red_teaming_prompt=red_teaming_orchestrator._initial_red_teaming_prompt,
                expected_red_teaming_responses=[fixed_input_prompt],
                expected_target_responses=expected_target_responses,
                stop_after_n_conversations=2,
                skip_initial_red_teaming_prompt_generation=True,
            )

            mock_rt.assert_not_called()
            mock_target.assert_called_once()

            second_fixed_input_prompt = "Second prompt to target - sent by user"
            expected_target_responses.append("Second target response")
            mock_target.return_value = expected_target_responses[1]
            with pytest.raises(ValueError):
                target_response = red_teaming_orchestrator.send_prompt(prompt=second_fixed_input_prompt)
            mock_rt.assert_not_called()


@pytest.mark.parametrize("attack_strategy_as_str", [True, False])
def test_reach_goal_after_two_turns_end_token(
    prompt_target: PromptTarget,
    chat_completion_engine: AzureOpenAIChat,
    simple_attack_strategy: AttackStrategy,
    memory: DuckDBMemory,
    attack_strategy_as_str: bool,
):
    attack_strategy: Union[str | AttackStrategy] = (
        str(simple_attack_strategy) if attack_strategy_as_str else simple_attack_strategy
    )

    red_teaming_orchestrator = EndTokenRedTeamingOrchestrator(
        red_teaming_chat=chat_completion_engine,
        memory=memory,
        attack_strategy=attack_strategy,
        initial_red_teaming_prompt="how can I help you?",
        prompt_target=prompt_target,
        verbose=True,
    )

    with patch.object(red_teaming_orchestrator._red_teaming_chat, "complete_chat") as mock_rt:
        with patch.object(red_teaming_orchestrator._prompt_target, "complete_chat") as mock_target:
            expected_red_teaming_responses = [
                "First red teaming chat response",
                "Second red teaming chat response with expected end token <|done|>",
            ]
            mock_rt.side_effect = expected_red_teaming_responses
            expected_target_response = "First target response"
            mock_target.return_value = expected_target_response
            target_response = red_teaming_orchestrator.apply_attack_strategy_until_completion()
            assert target_response == expected_target_response
            conversations = red_teaming_orchestrator._memory.get_all_memory(ConversationData)
            # Expecting three conversation threads (two with red teaming chat and one with prompt target)
            assert len(conversations) == 7, f"Expected 7 conversations, got {len(conversations)}"
            check_conversations(
                conversations,
                red_teaming_meta_prompt=str(attack_strategy),
                initial_red_teaming_prompt=red_teaming_orchestrator._initial_red_teaming_prompt,
                expected_red_teaming_responses=expected_red_teaming_responses,
                expected_target_responses=[expected_target_response],
                stop_after_n_conversations=3,
            )

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


class MockOneToManyPromptConverter(PromptConverter):
    def convert(self, prompts: list[str]) -> list[str]:
        return prompts + prompts

    def is_one_to_one_converter(self) -> bool:
        return False


@pytest.mark.parametrize("OrchestratorType", [ScoringRedTeamingOrchestrator, EndTokenRedTeamingOrchestrator])
def test_converters_raise_with_one_to_many(OrchestratorType):
    kwargs = {
        "attack_strategy": "Test strategy with end token <|done|>",
        "red_teaming_chat": Mock(),
        "prompt_target": Mock(),
        "memory": Mock(),
        "prompt_converters": [MockOneToManyPromptConverter()],
    }
    if OrchestratorType == ScoringRedTeamingOrchestrator:
        kwargs["scorer"] = Mock()
    with pytest.raises(ValueError):
        OrchestratorType(**kwargs)


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
