# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import random
from string import ascii_lowercase
from typing import MutableSequence, Sequence

import pytest
from unit.mocks import get_sample_conversation_entries, get_sample_conversations

from pyrit.memory import PromptMemoryEntry
from pyrit.models import PromptRequestPiece


@pytest.fixture
def sample_conversations() -> MutableSequence[PromptRequestPiece]:
    return get_sample_conversations()


@pytest.fixture
def sample_conversation_entries() -> Sequence[PromptMemoryEntry]:
    return get_sample_conversation_entries()


def generate_random_string(length: int = 10) -> str:
    return "".join(random.choice(ascii_lowercase) for _ in range(length))


def assert_original_value_in_list(original_value: str, prompt_request_pieces: Sequence[PromptRequestPiece]):
    for piece in prompt_request_pieces:
        if piece.original_value == original_value:
            return True
    raise AssertionError(f"Original value {original_value} not found in list")
