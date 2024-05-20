# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import pytest

from unittest.mock import MagicMock, patch
from tests.mocks import get_audio_request_piece, get_image_request_piece, get_test_request_piece

from pyrit.memory.memory_interface import MemoryInterface
from pyrit.models.prompt_request_piece import PromptRequestPiece
from pyrit.score.azure_content_filter_scorer import AzureContentFilterScorer
from azure.ai.contentsafety.models import TextCategory


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

    scorer = AzureContentFilterScorer(api_key="foo", endpoint="bar", harm_categories=TextCategory.HATE)
    with pytest.raises(ValueError, match="Azure Content Filter Scorer only supports text and image_path data type"):
        await scorer.validate(audio_request_piece)

    os.remove(audio_request_piece.converted_value)


@pytest.mark.asyncio
async def test_azure_content_filter_scorer_validate_image(image_request_piece: PromptRequestPiece):

    scorer = AzureContentFilterScorer(api_key="foo", endpoint="bar", harm_categories=TextCategory.HATE)

    # should not raise an error
    scorer.validate(image_request_piece)

    os.remove(image_request_piece.converted_value)


@pytest.mark.asyncio
async def test_azure_content_filter_scorer_validate_text(text_request_piece: PromptRequestPiece):

    scorer = AzureContentFilterScorer(api_key="foo", endpoint="bar", harm_categories=TextCategory.HATE)

    # should not raise an error
    scorer.validate(text_request_piece)


@pytest.mark.asyncio
async def test_azure_content_filter_scorer_get_azure_severity():

    scorer = AzureContentFilterScorer(api_key="foo", endpoint="bar", harm_categories=TextCategory.HATE)
    assert scorer.get_azure_severity("0.02") == 2


@pytest.mark.asyncio
@patch.object(AzureContentFilterScorer, "analyze_text", return_value=2)
async def test_azure_content_filter_scorer_adds_to_memory(analyze_text):
    memory = MagicMock(MemoryInterface)

    scorer = AzureContentFilterScorer(api_key="foo", endpoint="bar", harm_categories=TextCategory.HATE, memory=memory)
    await scorer.score_text_async(text="I hate you!")

    memory.add_scores_to_memory.assert_called_once()


@pytest.mark.asyncio
@patch.object(AzureContentFilterScorer, "analyze_text", return_value=2)
async def test_azure_content_filter_scorer_score(analyze_text):

    scorer = AzureContentFilterScorer(api_key="foo", endpoint="bar", harm_categories=TextCategory.HATE)
    score = await scorer.score_text_async(text="I hate you!")

    assert len(score) == 1

    assert score[0].score_value == "0.02"
    assert (
        score[0].score_value_description == "severity as float; use get_azure_severity to convert to int severity level"
    )
    assert score[0].score_type == "float_scale"
    assert score[0].score_category == str(TextCategory.HATE)
    assert score[0].score_metadata is None
    assert score[0].score_rationale is None
    assert "AzureContentFilterScorer" in str(score[0].scorer_class_identifier)
