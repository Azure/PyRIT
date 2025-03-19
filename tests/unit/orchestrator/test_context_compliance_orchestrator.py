# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from pyrit.models import SeedPrompt, SeedPromptDataset, SeedPromptGroup
from pyrit.orchestrator import ContextComplianceOrchestrator, ContextDescriptionPaths
from pyrit.prompt_converter import SearchReplaceConverter
from pyrit.prompt_normalizer import NormalizerRequest
from pyrit.prompt_target.common.prompt_chat_target import PromptChatTarget
from pyrit.score import Scorer


@pytest.fixture
def mock_objective_target():
    return MagicMock(spec=PromptChatTarget)


@pytest.fixture
def mock_adversarial_chat():
    return MagicMock(spec=PromptChatTarget)


@pytest.fixture
def mock_scorer():
    return MagicMock(spec=Scorer)


@pytest.fixture
def mock_seed_prompt_dataset():
    """
    Creates a mock of SeedPromptDataset with three prompts,
    matching the usage in ContextComplianceOrchestrator.
    """
    rephrase_user_mock = MagicMock(spec=SeedPrompt)
    rephrase_user_mock.render_template_value.return_value = "Mock rephrase to user"

    user_answer_mock = MagicMock(spec=SeedPrompt)
    user_answer_mock.render_template_value.return_value = "Mock user answer"

    rephrase_question_mock = MagicMock(spec=SeedPrompt)
    rephrase_question_mock.render_template_value.return_value = "Mock objective as question"

    dataset_mock = MagicMock(spec=SeedPromptDataset)
    dataset_mock.prompts = [rephrase_user_mock, user_answer_mock, rephrase_question_mock]
    return dataset_mock


@pytest.fixture
def context_compliance_orchestrator(
    mock_objective_target,
    mock_adversarial_chat,
    mock_scorer,
    mock_seed_prompt_dataset,
    patch_central_database,
):
    """
    Patches SeedPromptDataset.from_yaml_file so the orchestrator
    loads mock_seed_prompt_dataset by default.
    """
    with patch(
        "pyrit.orchestrator.single_turn.context_compliance_orchestrator.SeedPromptDataset.from_yaml_file",
        return_value=mock_seed_prompt_dataset,
    ):
        orchestrator = ContextComplianceOrchestrator(
            objective_target=mock_objective_target,
            adversarial_chat=mock_adversarial_chat,
            prompt_converters=None,  # Will default to the internal SearchReplaceConverter
            scorers=[mock_scorer],
            batch_size=5,
            verbose=True,
        )
    return orchestrator


def test_init(context_compliance_orchestrator, mock_seed_prompt_dataset):
    assert context_compliance_orchestrator._batch_size == 5
    assert context_compliance_orchestrator._verbose is True
    assert len(context_compliance_orchestrator._scorers) == 1

    # Check that the converter is a SearchReplaceConverter
    converters = context_compliance_orchestrator._prompt_converters
    assert len(converters) == 1
    assert isinstance(converters[0], SearchReplaceConverter)
    assert converters[0].pattern == r"^.*\Z"

    assert context_compliance_orchestrator._rephrase_objective_to_user_turn == mock_seed_prompt_dataset.prompts[0]
    assert context_compliance_orchestrator._answer_user_turn == mock_seed_prompt_dataset.prompts[1]
    assert context_compliance_orchestrator._rephrase_objective_to_question == mock_seed_prompt_dataset.prompts[2]


@pytest.mark.parametrize("context_path", list(ContextDescriptionPaths))
def test_context_description_paths_exist(context_path):
    """
    Checks the yaml files in ContextDescriptionPaths exist and are valid.
      1) Each path in ContextDescriptionPaths should exist as a file
      2) Loading it as a SeedPromptDataset should not error
      3) The dataset should have at least three prompts
    """
    path: Path = context_path.value
    assert path.is_file(), f"Path does not exist or is not a file: {path}"

    dataset = SeedPromptDataset.from_yaml_file(path)
    assert hasattr(dataset, "prompts"), "Dataset missing 'prompts' attribute"
    assert len(dataset.prompts) >= 3, "Expected at least 3 prompts in context description"


def test_validate_normalizer_requests(context_compliance_orchestrator):

    with pytest.raises(ValueError, match="No normalizer requests provided"):
        context_compliance_orchestrator.validate_normalizer_requests(prompt_request_list=[])

    multi_part_group = SeedPromptGroup(
        prompts=[SeedPrompt(value="test1", data_type="text"), SeedPrompt(value="test2", data_type="text")]
    )
    with pytest.raises(ValueError, match="Multi-part messages not supported"):
        context_compliance_orchestrator.validate_normalizer_requests(
            prompt_request_list=[NormalizerRequest(seed_prompt_group=multi_part_group)]
        )

    # Non-text prompt -> ValueError
    non_text_group = SeedPromptGroup(prompts=[SeedPrompt(value="image data", data_type="image")])
    with pytest.raises(ValueError, match="Non text messages not supported"):
        context_compliance_orchestrator.validate_normalizer_requests(
            prompt_request_list=[NormalizerRequest(seed_prompt_group=non_text_group)]
        )

    # Valid request -> no exception
    valid_group = SeedPromptGroup(prompts=[SeedPrompt(value="Hello world", data_type="text")])
    try:
        context_compliance_orchestrator.validate_normalizer_requests(
            prompt_request_list=[NormalizerRequest(seed_prompt_group=valid_group)]
        )
    except Exception as e:
        pytest.fail(f"Unexpected exception on valid request: {e}")


@pytest.mark.asyncio
async def test_get_prepended_conversation_async(context_compliance_orchestrator):
    from unittest.mock import AsyncMock, MagicMock

    mock_normalizer = MagicMock()
    # Use AsyncMock for async methods
    mock_normalizer.send_prompt_async = AsyncMock(
        side_effect=[
            MagicMock(get_value=lambda: "benign_user_query"),
            MagicMock(get_value=lambda: "some_user_answer"),
            MagicMock(get_value=lambda: "objective_as_question"),
        ]
    )

    context_compliance_orchestrator._prompt_normalizer = mock_normalizer

    seed_prompt_group = SeedPromptGroup(prompts=[SeedPrompt(value="Original objective", data_type="text")])
    normalizer_request = NormalizerRequest(seed_prompt_group=seed_prompt_group)

    result = await context_compliance_orchestrator.get_prepended_conversation_async(normalizer_request)
    assert len(result) == 2, "Should produce two PromptRequestResponse items: user turn + assistant turn"

    user_msg = result[0]
    assert user_msg.request_pieces[0].role == "user"
    assert user_msg.get_value() == "benign_user_query"

    # The assistant turn
    assistant_msg = result[1]
    assert assistant_msg.request_pieces[0].role == "assistant"
    # We expect the concatenation of the user query answer + objective question
    assert "some_user_answer" in assistant_msg.request_pieces[0].original_value
    assert "objective_as_question" in assistant_msg.request_pieces[0].original_value

    # Verify each underlying method was called
    assert mock_normalizer.send_prompt_async.call_count == 3
    call_args = mock_normalizer.send_prompt_async.call_args_list
    assert "Mock rephrase to user" == call_args[0].kwargs["seed_prompt_group"].prompts[0].value
    assert "Mock user answer" == call_args[1].kwargs["seed_prompt_group"].prompts[0].value
    assert "Mock objective as question" == call_args[2].kwargs["seed_prompt_group"].prompts[0].value
