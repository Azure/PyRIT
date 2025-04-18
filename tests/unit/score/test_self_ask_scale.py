# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import uuid
from pathlib import Path
from textwrap import dedent
from unittest.mock import AsyncMock, MagicMock

import pytest

from pyrit.models import PromptRequestPiece, PromptRequestResponse, UnvalidatedScore
from pyrit.score.self_ask_category_scorer import ContentClassifierPaths
from pyrit.score.self_ask_scale_scorer import SelfAskScaleScorer

tree_scale_path = SelfAskScaleScorer.ScalePaths.TREE_OF_ATTACKS_SCALE.value
task_scale_path = SelfAskScaleScorer.ScalePaths.TASK_ACHIEVED_SCALE.value
criteria_scale_path = SelfAskScaleScorer.ScalePaths.CRITERIA_SCALE.value
general_system_prompt_path = SelfAskScaleScorer.SystemPaths.GENERAL_SYSTEM_PROMPT.value
red_teamer_system_prompt_path = SelfAskScaleScorer.SystemPaths.RED_TEAMER_SYSTEM_PROMPT.value
criteria_system_prompt_path = SelfAskScaleScorer.SystemPaths.CRITERIA_SYSTEM_PROMPT.value


@pytest.fixture
def scorer_scale_response() -> PromptRequestResponse:

    json_response = (
        dedent(
            """
        {"score_value": "1",
         "rationale": "rationale",
         "description": "description"}
        """
        )
        .strip()
        .replace("\n", " ")
    )

    return PromptRequestResponse(request_pieces=[PromptRequestPiece(role="assistant", original_value=json_response)])


@pytest.fixture
def scale_scorer(patch_central_database) -> SelfAskScaleScorer:
    return SelfAskScaleScorer(
        chat_target=MagicMock(),
        scale_arguments_path=SelfAskScaleScorer.ScalePaths.TREE_OF_ATTACKS_SCALE.value,
        system_prompt_path=SelfAskScaleScorer.SystemPaths.GENERAL_SYSTEM_PROMPT.value,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "scale_arguments_path, system_prompt_path",
    [
        (None, None),
        (tree_scale_path, general_system_prompt_path),
        (task_scale_path, red_teamer_system_prompt_path),
        (tree_scale_path, red_teamer_system_prompt_path),
        (task_scale_path, general_system_prompt_path),
        (criteria_scale_path, criteria_system_prompt_path),
    ],
)
async def test_scale_scorer_set_system_prompt(
    scorer_scale_response: PromptRequestResponse,
    scale_arguments_path: Path,
    system_prompt_path: Path,
    patch_central_database,
):
    chat_target = MagicMock()
    chat_target.send_prompt_async = AsyncMock(return_value=scorer_scale_response)
    scorer = SelfAskScaleScorer(
        chat_target=chat_target,
        scale_arguments_path=scale_arguments_path,
        system_prompt_path=system_prompt_path,
    )

    await scorer.score_text_async(text="string", task="task")

    chat_target.set_system_prompt.assert_called_once()

    # assert that the scale score was loaded into system prompt

    assert scorer._system_prompt
    assert str(scorer._minimum_value) in scorer._system_prompt
    assert str(scorer._maximum_value) in scorer._system_prompt


def test_scale_scorer_invalid_scale_file_contents():
    chat_target = MagicMock()
    # When using a YAML with wrong keys the Scale constructor will raise an exception.
    with pytest.raises(ValueError, match="Missing key in scale_args:"):
        SelfAskScaleScorer(
            chat_target=chat_target,
            scale_arguments_path=ContentClassifierPaths.HARMFUL_CONTENT_CLASSIFIER.value,
            system_prompt_path=SelfAskScaleScorer.SystemPaths.GENERAL_SYSTEM_PROMPT.value,
        )


@pytest.mark.parametrize(
    "scale_args",
    [
        {
            "minimum_value": 0,
            "maximum_value": 1,
        },
        {
            "minimum_value": 0,
            "category": "category",
        },
        {
            "maximum_value": 1,
            "category": "category",
        },
        {
            "minimum_value": 0,
            "maximum_value": 1,
            "category": None,
        },
        {
            "minimum_value": "Blah",
            "maximum_value": 1,
            "category": "test",
        },
        {
            "minimum_value": 2,
            "maximum_value": 1,
            "category": "test",
        },
    ],
)
def test_validate_scale_arguments_missing_args_raises_value_error(scale_args, scale_scorer):
    with pytest.raises(ValueError):
        scale_scorer._validate_scale_arguments_set(scale_args)


@pytest.mark.asyncio
async def test_scale_scorer_score(scorer_scale_response: PromptRequestResponse, patch_central_database):

    chat_target = MagicMock()

    chat_target.send_prompt_async = AsyncMock(return_value=scorer_scale_response)
    scorer = SelfAskScaleScorer(
        chat_target=chat_target,
        scale_arguments_path=SelfAskScaleScorer.ScalePaths.TREE_OF_ATTACKS_SCALE.value,
        system_prompt_path=SelfAskScaleScorer.SystemPaths.GENERAL_SYSTEM_PROMPT.value,
    )

    score = await scorer.score_text_async(text="example text", task="task")

    assert len(score) == 1

    assert score[0].score_value == "0.0"
    assert score[0].get_value() == 0
    assert "description" in score[0].score_value_description
    assert "rationale" in score[0].score_rationale
    assert score[0].score_type == "float_scale"
    assert score[0].score_category == "jailbreak"
    assert score[0].prompt_request_response_id is None
    assert score[0].task == "task"


@pytest.mark.asyncio
async def test_scale_scorer_score_custom_scale(scorer_scale_response: PromptRequestResponse, patch_central_database):

    chat_target = MagicMock()

    # set a higher score to test the scaling
    scorer_scale_response.request_pieces[0].original_value = scorer_scale_response.request_pieces[
        0
    ].original_value.replace("1", "53")
    scorer_scale_response.request_pieces[0].converted_value = scorer_scale_response.request_pieces[0].original_value

    chat_target.send_prompt_async = AsyncMock(return_value=scorer_scale_response)
    scorer = SelfAskScaleScorer(
        chat_target=chat_target,
        scale_arguments_path=SelfAskScaleScorer.ScalePaths.TREE_OF_ATTACKS_SCALE.value,
        system_prompt_path=SelfAskScaleScorer.SystemPaths.GENERAL_SYSTEM_PROMPT.value,
    )

    scorer._minimum_value = 1
    scorer._maximum_value = 100

    score = await scorer.score_text_async(text="example text", task="task")

    assert len(score) == 1

    expected_score_value = (53 - 1) / (100 - 1)
    assert score[0].score_value == str(expected_score_value)
    assert score[0].get_value() == expected_score_value
    assert "description" in score[0].score_value_description
    assert "rationale" in score[0].score_rationale
    assert score[0].score_type == "float_scale"
    assert score[0].score_category == "jailbreak"
    assert score[0].prompt_request_response_id is None
    assert score[0].task == "task"


@pytest.mark.asyncio
async def test_scale_scorer_score_calls_send_chat(patch_central_database):

    chat_target = MagicMock()

    scorer = SelfAskScaleScorer(
        chat_target=chat_target,
        scale_arguments_path=SelfAskScaleScorer.ScalePaths.TREE_OF_ATTACKS_SCALE.value,
        system_prompt_path=SelfAskScaleScorer.SystemPaths.GENERAL_SYSTEM_PROMPT.value,
    )

    score = UnvalidatedScore(
        raw_score_value="1",
        score_rationale="rationale",
        score_type="float_scale",
        score_category="jailbreak",
        task="task",
        score_value_description="description",
        score_metadata="metadata",
        scorer_class_identifier="identifier",
        prompt_request_response_id=uuid.uuid4(),
    )

    score.prompt_request_response_id = None

    scorer._score_value_with_llm = AsyncMock(return_value=score)

    await scorer.score_text_async(text="example text", task="task")
    assert scorer._score_value_with_llm.call_count == int(1)
