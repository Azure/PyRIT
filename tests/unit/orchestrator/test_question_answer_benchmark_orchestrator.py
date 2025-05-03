# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from pyrit.models import (
    QuestionAnsweringEntry,
    QuestionChoice,
    SeedPrompt,
    SeedPromptGroup,
)
from pyrit.orchestrator import (
    PromptSendingOrchestrator,
    QuestionAnsweringBenchmarkOrchestrator,
)
from pyrit.prompt_target.common.prompt_chat_target import PromptChatTarget
from pyrit.score import Scorer


@pytest.fixture
def mock_objective_target() -> MagicMock:
    mock = MagicMock(spec=PromptChatTarget)
    mock._max_requests_per_minute = None
    return mock


@pytest.fixture
def mock_scorer():
    scorer = MagicMock(spec=Scorer)
    scorer.scorer_type = "true_false"
    return scorer


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
def question_answer_orchestrator(mock_objective_target, mock_scorer, patch_central_database):
    """Creates a QuestionAnsweringBenchmarkOrchestrator instance with mocked dependencies."""
    return QuestionAnsweringBenchmarkOrchestrator(
        objective_target=mock_objective_target,
        objective_scorer=mock_scorer,
        batch_size=3,
        verbose=True,
    )


def test_init(question_answer_orchestrator, mock_objective_target, mock_scorer):
    """Verifies that the orchestrator sets internal fields on init correctly."""
    assert question_answer_orchestrator._objective_target == mock_objective_target
    assert question_answer_orchestrator._objective_scorer == mock_scorer
    assert question_answer_orchestrator._batch_size == 3
    assert question_answer_orchestrator._verbose is True
    assert (
        question_answer_orchestrator._question_asking_format_string
        == QuestionAnsweringBenchmarkOrchestrator.QUESTION_ASKING_FORMAT_STRING
    )
    assert (
        question_answer_orchestrator._options_format_string
        == QuestionAnsweringBenchmarkOrchestrator.OPTIONS_FORMAT_STRING
    )
    assert (
        question_answer_orchestrator._objective_format_string
        == QuestionAnsweringBenchmarkOrchestrator.OBJECTIVE_FORMAT_STRING
    )


def test_get_objective(question_answer_orchestrator, mock_question_answer_entry):
    """Tests that _get_objective correctly formats the objective string."""
    objective = question_answer_orchestrator._get_objective(mock_question_answer_entry)
    expected = QuestionAnsweringBenchmarkOrchestrator.OBJECTIVE_FORMAT_STRING.format(
        question="What is the capital of France?", index="0", answer="Paris"
    )
    assert objective == expected


def test_get_objective_invalid_choice(question_answer_orchestrator):
    """Tests that _get_objective raises ValueError for invalid correct_answer."""
    invalid_entry = QuestionAnsweringEntry(
        question="What is the capital of France?",
        answer_type="str",
        correct_answer="4",  # Invalid index
        choices=[
            QuestionChoice(index=0, text="Paris"),
            QuestionChoice(index=1, text="London"),
        ],
    )
    with pytest.raises(ValueError, match="No matching choice found for correct_answer '4'"):
        question_answer_orchestrator._get_objective(invalid_entry)


def test_get_question_text(question_answer_orchestrator, mock_question_answer_entry):
    """Tests that _get_question_text correctly formats the question and options."""
    seed_prompt_group = question_answer_orchestrator._get_question_text(mock_question_answer_entry)

    assert isinstance(seed_prompt_group, SeedPromptGroup)
    assert len(seed_prompt_group.prompts) == 1

    prompt = seed_prompt_group.prompts[0]
    assert isinstance(prompt, SeedPrompt)
    assert prompt.data_type == "text"

    # Check that the formatted text contains the question and all options
    formatted_text = prompt.value
    assert "What is the capital of France?" in formatted_text
    assert "Option 0: Paris" in formatted_text
    assert "Option 1: London" in formatted_text
    assert "Option 2: Berlin" in formatted_text
    assert "Option 3: Madrid" in formatted_text


@pytest.mark.asyncio
async def test_run_attack_async(question_answer_orchestrator, mock_question_answer_entry):
    """Tests that run_attack_async properly formats the prompt and calls the parent class method."""
    with patch.object(PromptSendingOrchestrator, "run_attack_async", new_callable=AsyncMock) as mock_run_attack_async:
        mock_run_attack_async.return_value = MagicMock()

        await question_answer_orchestrator.run_attack_async(question_answering_entry=mock_question_answer_entry)

        # Verify the call to parent class method
        mock_run_attack_async.assert_called_once()
        call_kwargs = mock_run_attack_async.call_args.kwargs

        # Check the seed prompt group
        seed_prompt = call_kwargs["seed_prompt"]
        assert isinstance(seed_prompt, SeedPromptGroup)
        assert len(seed_prompt.prompts) == 1
        assert seed_prompt.prompts[0].data_type == "text"

        # Check that the formatted text contains the question and all options
        formatted_text = seed_prompt.prompts[0].value
        assert "What is the capital of France?" in formatted_text
        assert "Option 0: Paris" in formatted_text
        assert "Option 1: London" in formatted_text
        assert "Option 2: Berlin" in formatted_text
        assert "Option 3: Madrid" in formatted_text

        # Check the objective
        objective = call_kwargs["objective"]
        expected_objective = QuestionAnsweringBenchmarkOrchestrator.OBJECTIVE_FORMAT_STRING.format(
            question="What is the capital of France?", index="0", answer="Paris"
        )
        assert objective == expected_objective


@pytest.mark.asyncio
async def test_run_attacks_async(question_answer_orchestrator, mock_question_answer_entry):
    """Tests that run_attacks_async properly handles multiple entries."""
    entries = [mock_question_answer_entry, mock_question_answer_entry]

    with patch.object(
        QuestionAnsweringBenchmarkOrchestrator, "run_attack_async", new_callable=AsyncMock
    ) as mock_run_attack_async:
        mock_run_attack_async.return_value = MagicMock()

        results = await question_answer_orchestrator.run_attacks_async(question_answering_entries=entries)

        assert mock_run_attack_async.call_count == 2
        assert len(results) == 2
