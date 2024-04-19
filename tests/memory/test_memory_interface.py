# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Generator
import pytest
import random
from string import ascii_lowercase

from pyrit.memory import MemoryInterface
from pyrit.memory.memory_models import PromptRequestPiece

from tests.mocks import get_memory_interface, get_sample_conversations


@pytest.fixture
def memory_interface() -> Generator[MemoryInterface, None, None]:
    yield from get_memory_interface()


@pytest.fixture
def sample_conversations() -> list[PromptRequestPiece]:
    return get_sample_conversations()


def generate_random_string(length: int = 10) -> str:
    return "".join(random.choice(ascii_lowercase) for _ in range(length))


def test_memory(memory_interface: MemoryInterface):
    assert memory_interface


def test_conversation_memory_empty_by_default(memory_interface: MemoryInterface):
    expected_count = 0
    c = memory_interface.get_all_prompt_entries()
    assert len(c) == expected_count


@pytest.mark.parametrize("num_conversations", [1, 2, 3])
def test_add_request_pieces_to_memory(
    memory_interface: MemoryInterface, sample_conversations: list[PromptRequestPiece], num_conversations: int
):
    memory_interface.add_request_pieces_to_memory(request_pieces=sample_conversations[:num_conversations])
    assert len(memory_interface.get_all_prompt_entries()) == num_conversations
