# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import pytest

from pyrit.prompt_target.hugging_face.hugging_face_endpoint_target import (
    HuggingFaceEndpointTarget,
)


@pytest.fixture
def hugging_face_endpoint_target(patch_central_database) -> HuggingFaceEndpointTarget:
    return HuggingFaceEndpointTarget(
        hf_token="test_token",
        endpoint="https://api-inference.huggingface.co/models/test-model",
        model_id="test-model",
    )


def test_hugging_face_endpoint_initializes(hugging_face_endpoint_target: HuggingFaceEndpointTarget):
    assert hugging_face_endpoint_target


def test_hugging_face_endpoint_sets_endpoint_and_rate_limit():
    target = HuggingFaceEndpointTarget(
        hf_token="test_token",
        endpoint="https://api-inference.huggingface.co/models/test-model",
        model_id="test-model",
        max_requests_per_minute=30,
    )
    identifier = target.get_identifier()
    assert identifier["endpoint"] == "https://api-inference.huggingface.co/models/test-model"
    assert target._max_requests_per_minute == 30
