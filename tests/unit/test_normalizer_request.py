# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


import uuid

import pytest

from pyrit.models.seed_prompt import SeedPrompt, SeedPromptGroup
from pyrit.prompt_normalizer import NormalizerRequest


def test_normalizer_request_validates_sequence():
    group_id = str(uuid.uuid4())

    seed_prompt_group = SeedPromptGroup(
        prompts=[
            SeedPrompt(value="Hello", data_type="text", prompt_group_id=group_id, sequence=0),
            SeedPrompt(value="World", data_type="text", prompt_group_id=group_id, sequence=1),
        ]
    )

    request = NormalizerRequest(
        seed_prompt_group=seed_prompt_group,
        conversation_id=str(uuid.uuid4()),
    )

    with pytest.raises(ValueError) as exc_info:
        request.validate()

    assert "Sequence must be equal for every piece of a single normalizer request." in str(exc_info.value)


def test_normalizer_request_validates_empty_group():
    request = NormalizerRequest(
        seed_prompt_group=[],
        conversation_id=str(uuid.uuid4()),
    )

    with pytest.raises(ValueError) as exc_info:
        request.validate()

    assert "Seed prompt group must be provided." in str(exc_info.value)
