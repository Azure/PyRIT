# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
from unittest.mock import MagicMock, patch

import pytest
from azure.ai.contentsafety.models import TextCategory
from unit.mocks import (
    get_audio_request_piece,
    get_image_request_piece,
    get_test_request_piece,
)

from pyrit.memory import CentralMemory
from pyrit.memory.memory_interface import MemoryInterface
from pyrit.models import PromptRequestPiece
from pyrit.score.azure_content_filter_scorer import AzureContentFilterScorer


@pytest.fixture
def audio_request_piece() -> PromptRequestPiece:
    return get_audio_request_piece()


@pytest.fixture
def image_request_piece() -> PromptRequestPiece:
    return get_image_request_piece()


@pytest.fixture
def text_request_piece() -> PromptRequestPiece:
    return get_test_request_piece()


@pytest.mark.asyncio
async def test_azure_content_filter_scorer_validate_audio(audio_request_piece: PromptRequestPiece):
    scorer = AzureContentFilterScorer(api_key="foo", endpoint="bar", harm_categories=[TextCategory.HATE])
    with pytest.raises(ValueError, match="Azure Content Filter Scorer only supports text and image_path data type"):
        await scorer.validate(audio_request_piece)

    os.remove(audio_request_piece.converted_value)


@pytest.mark.asyncio
async def test_azure_content_filter_scorer_validate_image(image_request_piece: PromptRequestPiece):
    scorer = AzureContentFilterScorer(api_key="foo", endpoint="bar", harm_categories=[TextCategory.HATE])

    # should not raise an error
    scorer.validate(image_request_piece)

    os.remove(image_request_piece.converted_value)


@pytest.mark.asyncio
async def test_azure_content_filter_scorer_validate_text(text_request_piece: PromptRequestPiece):
    scorer = AzureContentFilterScorer(api_key="foo", endpoint="bar", harm_categories=[TextCategory.HATE])

    # should not raise an error
    scorer.validate(text_request_piece)


@pytest.mark.asyncio
async def test_azure_content_filter_scorer_adds_to_memory():
    memory = MagicMock(MemoryInterface)
    with patch.object(CentralMemory, "get_memory_instance", return_value=memory):

        scorer = AzureContentFilterScorer(api_key="foo", endpoint="bar", harm_categories=[TextCategory.HATE])

        mock_client = MagicMock()
        mock_client.analyze_text.return_value = {"categoriesAnalysis": [{"severity": "2", "category": "Hate"}]}

        scorer._azure_cf_client = mock_client

        await scorer.score_text_async(text="I hate you!")

        memory.add_scores_to_memory.assert_called_once()


@pytest.mark.asyncio
async def test_azure_content_filter_scorer_score(patch_central_database):
    scorer = AzureContentFilterScorer(api_key="foo", endpoint="bar", harm_categories=[TextCategory.HATE])

    mock_client = MagicMock()
    mock_client.analyze_text.return_value = {"categoriesAnalysis": [{"severity": "2", "category": "Hate"}]}

    scorer._azure_cf_client = mock_client

    score = await scorer.score_text_async(text="I hate you!")

    assert len(score) == 1

    # The azure score is on a ratio of 0-7, so 2/7 is the normalized value
    assert score[0].score_value == str(2.0 / 7)
    assert score[0].score_value_description is None
    assert score[0].score_type == "float_scale"
    assert score[0].score_category == str(TextCategory.HATE.value)
    assert score[0].score_metadata == str({"azure_severity": "2"})
    assert score[0].score_rationale is None
    assert "AzureContentFilterScorer" in str(score[0].scorer_class_identifier)


def test_azure_content_default_category():
    scorer = AzureContentFilterScorer(api_key="foo", endpoint="bar")
    assert len(scorer._score_categories) == 4


def test_azure_content_explicit_category():
    scorer = AzureContentFilterScorer(api_key="foo", endpoint="bar", harm_categories=[TextCategory.HATE])
    assert len(scorer._score_categories) == 1
