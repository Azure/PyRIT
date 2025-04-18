# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import pytest
from unit.mocks import MockPromptTarget
from unittest.mock import MagicMock, patch
from pyrit.prompt_target.common.prompt_chat_target import PromptChatTarget
from pyrit.score import Scorer
from pyrit.models import SeedPrompt, SeedPromptDataset

from pyrit.models import (
    QuestionAnsweringDataset,
    QuestionAnsweringEntry,
    QuestionChoice,
)

from pyrit.orchestrator import QuestionAnsweringBenchmarkOrchestrator

@pytest.fixture
def mock_objective_target(patch_central_database) -> MockPromptTarget:
    return MagicMock(spec=PromptChatTarget)

@pytest.fixture
def mock_prompt_converter():
    return MagicMock()

@pytest.fixture
def mock_scorer():
    return MagicMock(spec=Scorer)

@pytest.fixture
def mock_seed_prompt_dataset():
    """
    Creates a mock a SeedPromptDataset with two promtps corresponding to
    user_start_turn, assistant_start_turn
    """
    user_start_mock = MagicMock(spec=SeedPrompt)
    user_start_mock.value = "User start message"

    assistant_start_mock = MagicMock(spec=SeedPrompt)
    assistant_start_mock.value = "Assistant start message"

    dataset_mock = MagicMock(spec=SeedPromptDataset)
    dataset_mock.prompts = [user_start_mock, assistant_start_mock]

    return dataset_mock

@pytest.fixture
def question_answer_orchestrator(
    mock_objective_target,
    mock_prompt_converter,
    mock_scorer,
    mock_seed_prompt_dataset,
    patch_central_database,
):
    """
    A fixture that patches the from_yaml_file method so that
    `role_play_definition` loads the mock_seed_prompt_dataset.
    """
    with patch(
        "pyrit.orchestrator.single_turn.role_play_orchestrator.SeedPromptDataset.from_yaml_file",
        return_value=mock_seed_prompt_dataset,
    ):
        orchestrator = QuestionAnsweringBenchmarkOrchestrator(
            objective_target=mock_objective_target,
            question_answer_definition_path="fake/path/question_answer.yaml",
            prompt_converters=[mock_prompt_converter],
            scorers=[mock_scorer],
            batch_size=3,
            verbose=True,
        )
    return orchestrator

def test_init(question_answer_orchestrator, mock_seed_prompt_dataset):
    """
    Verifies that the orchestrator sets internal fields on init and loads
    the correct prompts from the given YAML dataset.
    """
    # Check if the orchestrator references the right rephrase/user/assistant prompts
    assert question_answer_orchestrator._user_start_turn == mock_seed_prompt_dataset.prompts[0]
    assert question_answer_orchestrator._assistant_start_turn == mock_seed_prompt_dataset.prompts[1]

    # Check batch size, verbosity, etc.
    assert question_answer_orchestrator._batch_size == 3
    assert question_answer_orchestrator._verbose is True