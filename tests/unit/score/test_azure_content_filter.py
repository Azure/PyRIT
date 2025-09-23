# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


import os
from unittest.mock import AsyncMock, MagicMock, patch

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
from pyrit.models.prompt_request_response import PromptRequestResponse
from pyrit.score.float_scale.azure_content_filter_scorer import AzureContentFilterScorer


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
async def test_score_piece_async_invalid_type(patch_central_database, audio_request_piece: PromptRequestPiece):
    scorer = AzureContentFilterScorer(api_key="foo", endpoint="bar", harm_categories=[TextCategory.HATE])
    request = PromptRequestResponse(
        request_pieces=[audio_request_piece],
    )

    # Should raise ValueError for unsupported data type
    with pytest.raises(ValueError, match="There are no valid pieces to score"):
        await scorer.score_async(request_response=request)
    os.remove(audio_request_piece.converted_value)


@pytest.mark.asyncio
async def test_score_piece_async_text(patch_central_database, text_request_piece: PromptRequestPiece):
    scorer = AzureContentFilterScorer(api_key="foo", endpoint="bar", harm_categories=[TextCategory.HATE])
    mock_client = MagicMock()
    mock_client.analyze_text.return_value = {"categoriesAnalysis": [{"severity": "2", "category": "Hate"}]}
    scorer._azure_cf_client = mock_client
    scores = await scorer._score_piece_async(text_request_piece)
    assert len(scores) == 1
    score = scores[0]
    assert score.score_type == "float_scale"
    assert score.score_value == str(2.0 / 7)
    assert score.score_category == ["Hate"]
    assert isinstance(score.score_metadata, dict)
    assert score.score_metadata["azure_severity"] == 2
    assert "AzureContentFilterScorer" in str(score.scorer_class_identifier)


@pytest.mark.asyncio
async def test_score_piece_async_image(patch_central_database, image_request_piece: PromptRequestPiece):
    scorer = AzureContentFilterScorer(api_key="foo", endpoint="bar", harm_categories=[TextCategory.HATE])
    mock_client = MagicMock()
    mock_client.analyze_image.return_value = {"categoriesAnalysis": [{"severity": "3", "category": "Hate"}]}
    scorer._azure_cf_client = mock_client
    # Patch _get_base64_image_data to avoid actual file IO
    with patch.object(scorer, "_get_base64_image_data", AsyncMock(return_value="base64data")):
        scores = await scorer._score_piece_async(image_request_piece)
    assert len(scores) == 1
    score = scores[0]
    assert score.score_type == "float_scale"
    assert score.score_value == str(3.0 / 7)
    assert score.score_category == ["Hate"]
    assert isinstance(score.score_metadata, dict)
    assert score.score_metadata["azure_severity"] == 3
    assert "AzureContentFilterScorer" in str(score.scorer_class_identifier)
    assert "AzureContentFilterScorer" in str(score.scorer_class_identifier)
    os.remove(image_request_piece.converted_value)


def test_default_category():
    scorer = AzureContentFilterScorer(api_key="foo", endpoint="bar")
    assert len(scorer._score_categories) == 4


def test_explicit_category():
    scorer = AzureContentFilterScorer(api_key="foo", endpoint="bar", harm_categories=[TextCategory.HATE])
    assert len(scorer._score_categories) == 1


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
    assert score[0].score_value_description == ""
    assert score[0].score_type == "float_scale"
    assert score[0].score_category == ["Hate"]
    assert score[0].score_metadata == {"azure_severity": 2}
    assert not score[0].score_rationale
    assert "AzureContentFilterScorer" in str(score[0].scorer_class_identifier)


def test_azure_content_default_category():
    scorer = AzureContentFilterScorer(api_key="foo", endpoint="bar")
    assert len(scorer._score_categories) == 4


def test_azure_content_explicit_category():
    scorer = AzureContentFilterScorer(api_key="foo", endpoint="bar", harm_categories=[TextCategory.HATE])
    assert len(scorer._score_categories) == 1
