# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
from unittest.mock import AsyncMock, MagicMock

import pytest

from pyrit.orchestrator.anecdoctor_orchestrator import AnecdoctorOrchestrator
from pyrit.prompt_normalizer import PromptNormalizer
from pyrit.prompt_target.common.prompt_chat_target import PromptChatTarget


@pytest.fixture
def mock_chat_model(patch_central_database) -> PromptChatTarget:
    return MagicMock(spec=PromptChatTarget)


@pytest.fixture
def mock_processing_model(patch_central_database) -> PromptChatTarget:
    return MagicMock(spec=PromptChatTarget)


@pytest.fixture
def example_data():
    """Example data for testing."""
    return [
        "Claim 1",
        "Claim 2",
        "Claim 3",
    ]


@pytest.fixture
def orchestrator(mock_chat_model, mock_processing_model, example_data):
    return AnecdoctorOrchestrator(
        chat_model_under_evaluation=mock_chat_model,
        use_knowledge_graph=False,
        processing_model=mock_processing_model,
        evaluation_data=example_data,
        language="german",
        content_type="news article",
        prompt_converters=[],
        verbose=False,
    )


def test_init(orchestrator, example_data):
    """Constructor sets internal state correctly."""
    assert orchestrator._evaluation_data == example_data
    assert orchestrator._language == "german"
    assert orchestrator._content_type == "news article"
    assert isinstance(orchestrator._normalizer, PromptNormalizer)
    assert orchestrator._use_knowledge_graph is False


@pytest.mark.asyncio
async def test_generate_attack_fewshot(monkeypatch, orchestrator):
    """The generate_attack() path without a KG should succeed and leave _kg_result None."""
    orchestrator._use_knowledge_graph = False

    # Patch YAML‑loader so no file access is required
    monkeypatch.setattr(
        AnecdoctorOrchestrator,
        "_load_prompt_from_yaml",
        lambda self, x: "Prompt in {language} for {type}",
    )

    # Stub normalizer response
    mock_response = MagicMock()
    mock_response.get_value.return_value = "Final output"
    orchestrator._normalizer.send_prompt_async = AsyncMock(return_value=mock_response)

    result = await orchestrator.generate_attack()
    assert orchestrator._kg_result is None
    assert result == "Final output"  # Validate the returned value


@pytest.mark.asyncio
async def test_generate_attack_with_kg(monkeypatch, orchestrator):
    """The generate_attack() flow with KG should call KG extraction and store the result."""
    orchestrator._use_knowledge_graph = True

    monkeypatch.setattr(
        AnecdoctorOrchestrator,
        "_load_prompt_from_yaml",
        lambda self, x: "Prompt in {language} for {type}",
    )

    orchestrator._extract_knowledge_graph = AsyncMock(return_value="Extracted KG")
    mock_response = MagicMock()
    mock_response.get_value.return_value = "Final with KG"
    orchestrator._normalizer.send_prompt_async = AsyncMock(return_value=mock_response)

    result = await orchestrator.generate_attack()
    assert orchestrator._kg_result == "Extracted KG"
    assert result == "Final with KG"  # Validate the returned value


@pytest.mark.asyncio
async def test_missing_evaluation_data(mock_chat_model, mock_processing_model):
    """Ensure ValueError is raised when evaluation_data is missing."""
    orchestrator = AnecdoctorOrchestrator(
        chat_model_under_evaluation=mock_chat_model,
        use_knowledge_graph=False,
        processing_model=mock_processing_model,
        evaluation_data=[],  # Empty data
        language="english",
        content_type="viral tweet",
    )

    with pytest.raises(ValueError, match="No example data provided for evaluation."):
        await orchestrator.generate_attack()


@pytest.mark.asyncio
async def test_missing_processing_model(mock_chat_model, example_data):
    """Ensure ValueError is raised when processing_model is missing and use_knowledge_graph=True."""
    orchestrator = AnecdoctorOrchestrator(
        chat_model_under_evaluation=mock_chat_model,
        use_knowledge_graph=True,
        processing_model=None,  # Missing processing model
        evaluation_data=example_data,
        language="english",
        content_type="viral tweet",
    )

    with pytest.raises(ValueError,
                       match="Processing model is not set. Cannot extract knowledge graph."):
        await orchestrator.generate_attack()
