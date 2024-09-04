# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
from textwrap import dedent
from typing import Generator
from unittest.mock import AsyncMock, MagicMock

import pytest
import yaml

from pyrit.exceptions.exception_classes import InvalidJsonException
from pyrit.memory.memory_interface import MemoryInterface
from pyrit.models import PromptRequestPiece
from pyrit.models import PromptRequestResponse
from pyrit.score import ScalePaths
from pyrit.score.self_ask_category_scorer import ContentClassifierPaths
from pyrit.score.self_ask_scale_scorer import Scale, ScaleExample, SelfAskScaleScorer

from tests.mocks import get_memory_interface


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
def memory() -> Generator[MemoryInterface, None, None]:
    yield from get_memory_interface()


@pytest.mark.asyncio
async def test_scale_scorer_set_system_prompt(scorer_scale_response: PromptRequestResponse):
    memory = MagicMock(MemoryInterface)
    chat_target = MagicMock()
    chat_target.send_prompt_async = AsyncMock(return_value=scorer_scale_response)

    scorer = SelfAskScaleScorer(
        chat_target=chat_target, scale_path=ScalePaths.TREE_OF_ATTACKS_WITH_PRUNING_SCALE.value, memory=memory
    )

    await scorer.score_text_async(text="string", task="task")

    chat_target.set_system_prompt.assert_called_once()

    # assert that the scale score was loaded into system prompt
    assert str(scorer._scale.category) in scorer._system_prompt


def test_scale_scorer_invalid_scale_file_contents():
    chat_target = MagicMock()
    # When using a YAML with wrong keys the Scale constructor will raise an exception.
    with pytest.raises(TypeError, match="unexpected keyword argument 'no_category_found'"):
        SelfAskScaleScorer(
            chat_target=chat_target,
            scale_path=ContentClassifierPaths.HARMFUL_CONTENT_CLASSIFIER.value,
        )


@pytest.mark.parametrize(
    "field_to_delete",
    [
        "category",
        "minimum_value",
        "maximum_value",
        "examples",
        "minimum_description",
        "maximum_description",
        "step_description",
    ],
)
def test_scale_must_have_fields(field_to_delete):
    scale_path = ScalePaths.TREE_OF_ATTACKS_WITH_PRUNING_SCALE.value
    scale_args = yaml.safe_load(scale_path.read_text(encoding="utf-8"))
    scale_args["examples"] = [
        ScaleExample(
            task=example["task"],
            response=example["response"],
            rationale=example["rationale"],
            score_value=example["score_value"],
        )
        for example in scale_args["examples"]
    ]
    del scale_args[field_to_delete]

    with pytest.raises(TypeError, match=f"required keyword-only argument: '{field_to_delete}'"):
        Scale(**scale_args)


def test_scale_invalid_min_max():
    scale_path = ScalePaths.TREE_OF_ATTACKS_WITH_PRUNING_SCALE.value
    scale_args = yaml.safe_load(scale_path.read_text(encoding="utf-8"))
    scale_args["examples"] = [
        ScaleExample(
            task=example["task"],
            response=example["response"],
            rationale=example["rationale"],
            score_value=example["score_value"],
        )
        for example in scale_args["examples"]
    ]
    scale_args["minimum_value"] = scale_args["maximum_value"] + 1

    with pytest.raises(ValueError, match="minimum_value must be less than maximum_value"):
        Scale(**scale_args)


def test_scale_no_examples():
    scale_path = ScalePaths.TREE_OF_ATTACKS_WITH_PRUNING_SCALE.value
    scale_args = yaml.safe_load(scale_path.read_text(encoding="utf-8"))
    scale_args["examples"] = []

    with pytest.raises(ValueError, match="examples must be provided"):
        Scale(**scale_args)


def test_scale_malformed_examples():
    scale_path = ScalePaths.TREE_OF_ATTACKS_WITH_PRUNING_SCALE.value
    scale_args = yaml.safe_load(scale_path.read_text(encoding="utf-8"))
    scale_args["examples"] = [{"wrong": "entry"}]

    with pytest.raises(ValueError, match="All examples must be of type ScaleExample"):
        Scale(**scale_args)


@pytest.mark.asyncio
async def test_scale_scorer_adds_to_memory(scorer_scale_response: PromptRequestResponse):
    memory = MagicMock(MemoryInterface)
    chat_target = MagicMock()
    chat_target.send_prompt_async = AsyncMock(return_value=scorer_scale_response)

    scorer = SelfAskScaleScorer(
        chat_target=chat_target, scale_path=ScalePaths.TREE_OF_ATTACKS_WITH_PRUNING_SCALE.value, memory=memory
    )

    await scorer.score_text_async(text="string", task="task")

    memory.add_scores_to_memory.assert_called_once()


@pytest.mark.asyncio
async def test_scale_scorer_score(memory: MemoryInterface, scorer_scale_response: PromptRequestResponse):

    chat_target = MagicMock()

    chat_target.send_prompt_async = AsyncMock(return_value=scorer_scale_response)

    scorer = SelfAskScaleScorer(
        chat_target=chat_target, scale_path=ScalePaths.TREE_OF_ATTACKS_WITH_PRUNING_SCALE.value, memory=memory
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
async def test_scale_scorer_score_custom_scale(memory: MemoryInterface, scorer_scale_response: PromptRequestResponse):

    chat_target = MagicMock()

    # set a higher score to test the scaling
    scorer_scale_response.request_pieces[0].original_value = scorer_scale_response.request_pieces[
        0
    ].original_value.replace("1", "53")
    scorer_scale_response.request_pieces[0].converted_value = scorer_scale_response.request_pieces[0].original_value

    chat_target.send_prompt_async = AsyncMock(return_value=scorer_scale_response)

    scorer = SelfAskScaleScorer(
        chat_target=chat_target,
        scale=Scale(
            category="jailbreak",
            minimum_value=1,
            maximum_value=100,
            minimum_description="minimum_description",
            maximum_description="maximum_description",
            step_description="step_description",
            examples=[
                ScaleExample(
                    task="task",
                    response="response",
                    rationale="rationale",
                    score_value=47,
                )
            ],
        ),
        memory=memory,
    )

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
async def test_self_ask_scorer_bad_json_exception_retries():

    chat_target = MagicMock()

    bad_json_resp = PromptRequestResponse(
        request_pieces=[PromptRequestPiece(role="assistant", original_value="this is not a json")]
    )
    chat_target.send_prompt_async = AsyncMock(return_value=bad_json_resp)

    scorer = SelfAskScaleScorer(
        chat_target=chat_target, scale_path=ScalePaths.TREE_OF_ATTACKS_WITH_PRUNING_SCALE.value, memory=memory
    )

    with pytest.raises(InvalidJsonException):
        await scorer.score_text_async(text="this has no bullying", task="task")
        assert chat_target.send_prompt_async.call_count == int(os.getenv("RETRY_MAX_NUM_ATTEMPTS"))


@pytest.mark.asyncio
async def test_self_ask_scale_scorer_json_missing_key_exception_retries():

    chat_target = MagicMock()

    json_response = (
        dedent(
            """
            {"score_value": "1",
            "wrongly_named_description": "No cyber harm. The output contains no code or instructions that could be
            used to exploit vulnerabilities or cause harm to computer systems and networks.",
            "rationale": "The message does not contain any code or instructions that could be used to cause harm
            to computer systems or networks. It is simply a statement expressing a conspiracy theory, which may be false
            but does not pose a direct threat to cybersecurity. Therefore, the level of cyber harm is 1."}
            """
        )
        .strip()
        .replace("\n", " ")
    )

    bad_json_resp = PromptRequestResponse(
        request_pieces=[PromptRequestPiece(role="assistant", original_value=json_response)]
    )

    chat_target.send_prompt_async = AsyncMock(return_value=bad_json_resp)

    scorer = SelfAskScaleScorer(
        chat_target=chat_target, scale_path=ScalePaths.TREE_OF_ATTACKS_WITH_PRUNING_SCALE.value, memory=memory
    )

    with pytest.raises(InvalidJsonException):
        await scorer.score_text_async(text="this has no bullying", task="task")
        assert chat_target.send_prompt_async.call_count == int(os.getenv("RETRY_MAX_NUM_ATTEMPTS"))
