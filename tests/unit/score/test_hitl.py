# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import csv
import os
import tempfile
from textwrap import dedent
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from unit.mocks import get_image_request_piece

from pyrit.memory import CentralMemory
from pyrit.memory.memory_interface import MemoryInterface
from pyrit.models import PromptRequestPiece, Score
from pyrit.score.human_in_the_loop_scorer import HumanInTheLoopScorer
from pyrit.score.substring_scorer import SubStringScorer


@pytest.fixture
def image_request_piece() -> PromptRequestPiece:
    return get_image_request_piece()


@pytest.fixture
def score() -> Score:
    return Score(
        score_value="True",
        score_value_description="description hi",
        score_type="true_false",
        score_category="category",
        score_rationale="rationale for this score",
        score_metadata="",
        prompt_request_response_id="",
    )


def test_import_scores_from_csv(score: Score):
    scorer = HumanInTheLoopScorer()

    memory = MagicMock(MemoryInterface)
    with patch.object(CentralMemory, "get_memory_instance", return_value=memory):

        field_names = [
            "score_value",
            "score_value_description",
            "score_type",
            "score_category",
            "score_rationale",
            "score_metadata",
            "scorer_class_identifier",
            "prompt_request_response_id",
        ]

        with tempfile.NamedTemporaryFile(mode="w+", delete=False) as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(field_names)
            csvwriter.writerow(
                [
                    score.score_value,
                    score.score_value_description,
                    score.score_type,
                    score.score_category,
                    score.score_rationale,
                    score.score_metadata,
                    score.scorer_class_identifier,
                    score.prompt_request_response_id,
                ]
            )

        scores = scorer.import_scores_from_csv(csv_file_path=csvfile.name)
        assert scores
        assert len(scores) == 1
        assert scores[0].score_value == score.score_value
        assert scores[0].score_value_description == score.score_value_description
        assert scores[0].score_type == score.score_type
        assert scores[0].score_category == score.score_category
        assert scores[0].score_rationale == score.score_rationale
        assert scores[0].score_metadata == score.score_metadata
        assert scores[0].scorer_class_identifier
        assert scores[0].scorer_class_identifier["__type__"] == "HumanInTheLoopScorer"
        assert scores[0].prompt_request_response_id == score.prompt_request_response_id

        memory.add_scores_to_memory.assert_called_once()
        os.remove(csvfile.name)


@pytest.mark.asyncio
async def test_substring_scorer_validate(image_request_piece: PromptRequestPiece):
    scorer = SubStringScorer(substring="test", category="new_category")
    with pytest.raises(ValueError, match="Expected text data type"):
        await scorer.score_async(image_request_piece)

    os.remove(image_request_piece.converted_value)


@pytest.mark.asyncio
@pytest.mark.parametrize("sub_string", ["I can't answer", "Not present"])
async def test_substring_scorer_score(patch_central_database, sub_string: str):
    full_text = "blah I can't answer that too"
    scorer = SubStringScorer(substring=sub_string, category="new_category")

    score = await scorer.score_text_async(full_text)

    assert len(score) == 1

    assert score[0].score_value == str(sub_string in full_text)
    assert score[0].score_type == "true_false"
    assert score[0].score_category == "new_category"
    assert score[0].prompt_request_response_id is None


@pytest.mark.asyncio
async def test_substring_scorer_adds_to_memory():
    memory = MagicMock(MemoryInterface)
    with patch.object(CentralMemory, "get_memory_instance", return_value=memory):
        scorer = SubStringScorer(substring="string", category="new_category")
        await scorer.score_text_async(text="string")

    memory.add_scores_to_memory.assert_called_once()


@pytest.fixture
def scorer_category_response_bullying() -> PromptRequestPiece:

    json_response = (
        dedent(
            """
        {"category": "bullying", "category_description": "This is bullying.", "rationale": "The message
        seems like it contains bullying.", "score_value": "True"}
        """
        )
        .strip()
        .replace("\n", " ")
    )

    return PromptRequestPiece(role="assistant", original_value=json_response, converted_value=json_response)


@pytest.mark.asyncio
@patch("tkinter.simpledialog.askstring", side_effect=["1"])
@patch("tkinter.Tk", MagicMock())  # Mock Tkinter root window
async def test_scorer_keep_score(mock_askstring, scorer_category_response_bullying: PromptRequestPiece):
    scorer = MagicMock()
    mock_score = Score(
        score_value="0.8",
        score_value_description="Bullying score",
        score_type="float_scale",
        score_category="bullying",
        score_rationale="Identified as harmful",
        score_metadata="metadata example",
        scorer_class_identifier={"score_name": "HumanInTheLoopScorer"},
        prompt_request_response_id="test_id",
    )
    scorer.score_async = AsyncMock(return_value=[mock_score])
    with patch.object(CentralMemory, "get_memory_instance", return_value=MagicMock()):
        # Initialize HumanInTheLoopScorer
        hitl_scorer = HumanInTheLoopScorer(scorer=scorer, re_scorers=[scorer])

        # Test the scoring
        scores = await hitl_scorer.score_async(scorer_category_response_bullying)

        # Assertions
        assert len(scores) == 1
        assert scores[0].score_value == "0.8"  # Ensure score is kept as per input "1"
