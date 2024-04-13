# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Generator
import pytest
import random
from string import ascii_lowercase

from pyrit.memory import MemoryInterface
from pyrit.memory.memory_models import PromptMemoryEntry
from pyrit.models.models import ChatMessage

from tests.mocks import get_memory_interface, get_sample_conversation_entries


@pytest.fixture
def memory_interface() -> Generator[MemoryInterface, None, None]:
    yield from get_memory_interface()

@pytest.fixture
def sample_entries() -> list[PromptMemoryEntry]:
    return get_sample_conversation_entries()


def generate_random_string(length: int = 10) -> str:
    return "".join(random.choice(ascii_lowercase) for _ in range(length))


def test_memory(memory_interface: MemoryInterface):
    assert memory_interface


def test_conversation_memory_empty_by_default(memory_interface: MemoryInterface):
    expected_count = 0
    c = memory_interface.get_all_prompt_entries()
    assert len(c) == expected_count


def test_count_of_memories_matches_number_of_conversations_added_1(
    memory_interface: MemoryInterface,
    sample_entries: list[PromptMemoryEntry]
):
    expected_count = 1
    message = sample_entries[0]
    memory_interface.insert_prompt_entries(entries=[message])
    c = memory_interface.get_all_prompt_entries()
    assert len(c) == expected_count


def test_insert_prompt_entries_added(
        memory_interface: MemoryInterface,
        sample_entries: list[PromptMemoryEntry]
):
    expected_count = 3
    memory_interface.insert_prompt_entries(entries=sample_entries)
    assert len(memory_interface.get_all_prompt_entries()) == expected_count
