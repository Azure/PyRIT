# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Generator
from unittest.mock import MagicMock, patch
import pytest

from pyrit.memory import CentralMemory
from pyrit.memory.memory_interface import MemoryInterface
from pyrit.score.markdown_injection import MarkdownInjectionScorer
from tests.mocks import get_memory_interface


@pytest.fixture
def memory() -> Generator[MemoryInterface, None, None]:
    yield from get_memory_interface()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "text",
    [
        "![sdf](www.google.com)",
        "![Drag Racing](Dragster.jpg)",
        "![][image_ref_a32ff4ads]",
    ],
)
async def test_md_injection_detected(text: str, memory: MemoryInterface):
    with patch.object(CentralMemory, "get_memory_instance", return_value=memory):
        scorer = MarkdownInjectionScorer()
        score = (await scorer.score_text_async(text))[0]
        assert score.get_value() is True


@pytest.mark.asyncio
async def test_md_injection_not_detected(memory: MemoryInterface):
    with patch.object(CentralMemory, "get_memory_instance", return_value=memory):
        classifier = MarkdownInjectionScorer()
        text = "this is a test"
        score = await classifier.score_text_async(text)
        assert score[0].get_value() is False


@pytest.mark.asyncio
async def test_md_injection_adds_to_memory():
    memory = MagicMock(MemoryInterface)
    with patch.object(CentralMemory, "get_memory_instance", return_value=memory):
        scorer = MarkdownInjectionScorer()
        await scorer.score_text_async(text="string")

        memory.add_scores_to_memory.assert_called_once()
