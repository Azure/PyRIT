# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from pyrit.models import SeedPrompt, SeedPromptDataset
from pyrit.orchestrator import RolePlayOrchestrator
from pyrit.orchestrator.single_turn.role_play_orchestrator import RolePlayPaths
from pyrit.prompt_converter import LLMGenericTextConverter
from pyrit.prompt_target.common.prompt_chat_target import PromptChatTarget
from pyrit.score import Scorer


@pytest.fixture
def mock_objective_target():
    return MagicMock(spec=PromptChatTarget)


@pytest.fixture
def mock_adversarial_chat():
    return MagicMock(spec=PromptChatTarget)


@pytest.fixture
def mock_prompt_converter():
    # Example converter mock if needed
    return MagicMock()


@pytest.fixture
def mock_scorer():
    # Example scorer mock if needed
    scorer = MagicMock(spec=Scorer)
    scorer.scorer_type = "true_false"  # Add the required scorer_type attribute
    return scorer


@pytest.fixture
def mock_seed_prompt_dataset():
    """
    Creates a mock of SeedPromptDataset with three prompts, corresponding to
    rephrase_instructions, user_start_turn, assistant_start_turn.
    """
    rephrase_mock = MagicMock(spec=SeedPrompt)
    # Pretend the template returns the objective with some special text
    rephrase_mock.parameters = {"objective": "Objective 1"}
    rephrase_mock.render_template_value.return_value = "Rephrased objective"

    user_start_mock = MagicMock(spec=SeedPrompt)
    user_start_mock.value = "User start message"

    assistant_start_mock = MagicMock(spec=SeedPrompt)
    assistant_start_mock.value = "Assistant start message"

    dataset_mock = MagicMock(spec=SeedPromptDataset)
    dataset_mock.prompts = [rephrase_mock, user_start_mock, assistant_start_mock]

    return dataset_mock


@pytest.fixture
def role_play_orchestrator(
    mock_objective_target,
    mock_adversarial_chat,
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
        orchestrator = RolePlayOrchestrator(
            objective_target=mock_objective_target,
            adversarial_chat=mock_adversarial_chat,
            role_play_definition_path="fake/path/role_play.yaml",
            request_converter_configurations=[mock_prompt_converter],
            objective_scorer=mock_scorer,
            batch_size=3,
            verbose=True,
        )
    return orchestrator


def test_init(role_play_orchestrator, mock_seed_prompt_dataset):
    """
    Verifies that the orchestrator sets internal fields on init and loads
    the correct prompts from the given YAML dataset.
    """
    # Check if the orchestrator references the right rephrase/user/assistant prompts
    assert role_play_orchestrator._rephrase_instructions == mock_seed_prompt_dataset.prompts[0]
    assert role_play_orchestrator._user_start_turn == mock_seed_prompt_dataset.prompts[1]
    assert role_play_orchestrator._assistant_start_turn == mock_seed_prompt_dataset.prompts[2]

    # Check batch size, verbosity, etc.
    assert role_play_orchestrator._batch_size == 3
    assert role_play_orchestrator._verbose is True


@pytest.mark.asyncio
async def test_get_conversation_start(role_play_orchestrator):
    """
    Ensures that the conversation start is correctly formatted with user and assistant messages.
    """
    conversation_start = await role_play_orchestrator._get_conversation_start()

    assert len(conversation_start) == 2, "Should have user and assistant start turns"
    assert conversation_start[0].request_pieces[0].role == "user"
    assert conversation_start[0].request_pieces[0].original_value == "User start message"
    assert conversation_start[1].request_pieces[0].role == "assistant"
    assert conversation_start[1].request_pieces[0].original_value == "Assistant start message"


@pytest.mark.asyncio
async def test_get_role_playing_sets_default_converter(role_play_orchestrator):
    """
    Ensures role playing orchestrator sets the default converter
    """
    assert len(role_play_orchestrator._request_converter_configurations) == 2
    assert isinstance(
        role_play_orchestrator._request_converter_configurations[0].converters[0], LLMGenericTextConverter
    )
    instructions = (
        role_play_orchestrator._request_converter_configurations[0]
        .converters[0]
        ._user_prompt_template_with_objective.render_template_value()
    )
    assert instructions == "Rephrased objective"


@pytest.mark.parametrize("role_play_path", list(RolePlayPaths))
def test_role_play_paths(role_play_path) -> None:
    """
    For each path in RolePlayPaths, verify that:
      1) The file actually exists on disk.
      2) The YAML loads into a SeedPromptDataset without error.
      3) The dataset has exactly three prompts.
    """
    path: Path = role_play_path.value
    assert path.is_file(), f"Path does not exist or is not a file: {path}"

    dataset = SeedPromptDataset.from_yaml_file(path)
    assert hasattr(dataset, "prompts"), f"File {path} didn't load a dataset with 'prompts'"
    assert len(dataset.prompts) == 3, f"Expected 3 prompts in {path}, found {len(dataset.prompts)}"
    assert "objective" in dataset.prompts[0].parameters, "The first prompt should have an 'objective' parameter"
