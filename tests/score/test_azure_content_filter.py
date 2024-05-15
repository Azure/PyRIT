# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import pytest

from pyrit.models.prompt_request_piece import PromptRequestPiece
from pyrit.score.azure_content_filter_scorer import AzureContentFilterScorer
from tests.mocks import get_audio_request_piece, get_image_request_piece, get_test_request_piece
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

    scorer = AzureContentFilterScorer(harm_category=TextCategory.HATE)
    with pytest.raises(ValueError, match="Azure Content Filter Scorer only supports text and image_path data type"):
        await scorer.validate(audio_request_piece)

    os.remove(audio_request_piece.converted_value)


@pytest.mark.asyncio
async def test_azure_content_filter_scorer_validate_image(image_request_piece: PromptRequestPiece):

    scorer = AzureContentFilterScorer(harm_category=TextCategory.HATE)

    # should not raise an error
    scorer.validate(image_request_piece)

    os.remove(image_request_piece.converted_value)


@pytest.mark.asyncio
async def test_azure_content_filter_scorer_validate_text(text_request_piece: PromptRequestPiece):

    scorer = AzureContentFilterScorer(harm_category=TextCategory.HATE)

    # should not raise an error
    scorer.validate(text_request_piece)
