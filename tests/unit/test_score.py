# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import uuid
from datetime import datetime

import pytest

from pyrit.models.score import Score


@pytest.mark.asyncio
async def test_score_to_dict():
    sample_score = Score(
        id=str(uuid.uuid4()),
        score_value="false",
        score_value_description="true false score",
        score_type="true_false",
        score_category="Category1",
        score_rationale="Rationale text",
        score_metadata={"key": "value"},
        scorer_class_identifier="Scorer1",
        prompt_request_response_id=str(uuid.uuid4()),
        timestamp=datetime.now(),
        task="Task1",
    )
    result = sample_score.to_dict()

    # Check that all keys are present
    expected_keys = [
        "id",
        "score_value",
        "score_value_description",
        "score_type",
        "score_category",
        "score_rationale",
        "score_metadata",
        "scorer_class_identifier",
        "prompt_request_response_id",
        "timestamp",
        "task",
    ]

    for key in expected_keys:
        assert key in result, f"Missing key: {key}"

    # Check the key values
    assert result["id"] == str(sample_score.id)
    assert result["score_value"] == sample_score.score_value
    assert result["score_value_description"] == sample_score.score_value_description
    assert result["score_type"] == sample_score.score_type
    assert result["score_category"] == sample_score.score_category
    assert result["score_rationale"] == sample_score.score_rationale
    assert result["score_metadata"] == sample_score.score_metadata
    assert result["scorer_class_identifier"] == sample_score.scorer_class_identifier
    assert result["prompt_request_response_id"] == str(sample_score.prompt_request_response_id)
    assert result["timestamp"] == sample_score.timestamp.isoformat()
    assert result["task"] == sample_score.task
