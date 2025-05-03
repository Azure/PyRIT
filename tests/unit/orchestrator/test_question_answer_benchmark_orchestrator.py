# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from unit.mocks import MockPromptTarget

from pyrit.models import SeedPrompt, SeedPromptDataset, SeedPromptGroup
from pyrit.orchestrator import QuestionAnsweringBenchmarkOrchestrator, PromptSendingOrchestrator
from pyrit.prompt_target.common.prompt_chat_target import PromptChatTarget
from pyrit.score import Scorer


@pytest.fixture
def mock_objective_target(patch_central_database) -> MockPromptTarget:
    return MagicMock(spec=PromptChatTarget)


@pytest.fixture
def mock_scorer():
    return MagicMock(spec=Scorer)


@pytest.fixture
def mock_question_answer_entry():
    return QuestionAnsweringEntry(
        question="What is the capital of France?",
        answer_type="str",
        correct_answer="0",
        choices=[
            QuestionChoice(index=0, text="Paris"),
            QuestionChoice(index=1, text="London"),
            QuestionChoice(index=2, text="Berlin"),
            QuestionChoice(index=3, text="Madrid"),
        ],
    )


@pytest.fixture
def question_answer_orchestrator(
    mock_objective_target,
    mock_scorer,
    mock_seed_prompt_dataset,
    patch_central_database,
):
    """
    A fixture that patches the from_yaml_file method so that
    `question_answer_definition` loads the mock_seed_prompt_dataset.
    """
    with patch(
        "pyrit.orchestrator.single_turn.question_answer_benchmark_orchestrator.SeedPromptDataset.from_yaml_file",
        return_value=mock_seed_prompt_dataset,
    ):
        orchestrator = QuestionAnsweringBenchmarkOrchestrator(
            objective_target=mock_objective_target,
            question_answer_definition_path="fake/path/question_answer.yaml",
            objective_scorer=mock_scorer,
            batch_size=3,
            verbose=True,
        )
    return orchestrator


def test_init(question_answer_orchestrator, mock_seed_prompt_dataset):
    """
    Verifies that the orchestrator sets internal fields on init and loads
    the correct prompts from the given YAML dataset.
    """
    # Check if the orchestrator references the right user/assistant prompts
    assert question_answer_orchestrator._user_start_turn == mock_seed_prompt_dataset.prompts[0]
    assert question_answer_orchestrator._assistant_start_turn == mock_seed_prompt_dataset.prompts[1]

    # Check batch size, verbosity, etc.
    assert question_answer_orchestrator._batch_size == 3
    assert question_answer_orchestrator._verbose is True


def test_failed_init_too_few_samples(
    mock_objective_target,
    mock_scorer,
    mock_single_prompt_seed_prompt_dataset,
    patch_central_database,
):
    """
    Verifies that constructing the orchestrator results in the expected ValueError
    """
    with patch(
        "pyrit.orchestrator.single_turn.question_answer_benchmark_orchestrator.SeedPromptDataset.from_yaml_file",
        return_value=mock_single_prompt_seed_prompt_dataset,
    ):
        with pytest.raises(ValueError, match=r"Prompt list must have exactly 2 elements \(user and assistant turns\)"):
            QuestionAnsweringBenchmarkOrchestrator(
                objective_target=mock_objective_target,
                question_answer_definition_path="fake/path/question_answer.yaml",
                objective_scorer=mock_scorer,
                batch_size=3,
                verbose=True,
            )


def test_failed_init_too_many_samples(
    mock_objective_target,
    mock_scorer,
    mock_three_prompt_seed_prompt_dataset,
    patch_central_database,
):
    """
    Verifies that constructing the orchestrator results in the expected ValueError
    """
    with patch(
        "pyrit.orchestrator.single_turn.question_answer_benchmark_orchestrator.SeedPromptDataset.from_yaml_file",
        return_value=mock_three_prompt_seed_prompt_dataset,
    ):
        with pytest.raises(ValueError, match=r"Prompt list must have exactly 2 elements \(user and assistant turns\)"):
            QuestionAnsweringBenchmarkOrchestrator(
                objective_target=mock_objective_target,
                question_answer_definition_path="fake/path/question_answer.yaml",
                objective_scorer=mock_scorer,
                batch_size=3,
                verbose=True,
            )


@pytest.mark.asyncio
async def test_run_attack_async(question_answer_orchestrator):
    """
    Tests that run_attack_async properly formats the prompt and calls the parent class method
    """
    objective = "What is 1 + 1?"

    with patch.object(
        PromptSendingOrchestrator, "run_attack_async", new_callable=AsyncMock
    ) as mock_run_attack_async:
        mock_run_attack_async.return_value = MagicMock()

        await question_answer_orchestrator.run_attack_async(objective=objective)

        # Verify the call to parent class method
        mock_run_attack_async.assert_called_once()
        call_kwargs = mock_run_attack_async.call_args.kwargs
        
        # Check the seed prompt group
        seed_prompt = call_kwargs["seed_prompt"]
        assert isinstance(seed_prompt, SeedPromptGroup)
        assert len(seed_prompt.prompts) == 1
        assert seed_prompt.prompts[0].value == objective
        assert seed_prompt.prompts[0].data_type == "text"

        # Check the prepended conversation
        prepended = call_kwargs["prepended_conversation"]
        assert len(prepended) == 2
        assert prepended[0].request_pieces[0].role == "user"
        assert prepended[1].request_pieces[0].role == "assistant"
        assert "You are a helpful AI assistant" in prepended[0].request_pieces[0].original_value
        assert "I will help you" in prepended[1].request_pieces[0].original_value
