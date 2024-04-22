# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import tempfile
import pytest

from pyrit.memory import DuckDBMemory
from pyrit.orchestrator import PromptSendingOrchestrator
from pyrit.prompt_converter import Base64Converter, StringJoinConverter

from tests.mocks import MockPromptTarget


@pytest.fixture
def mock_target() -> MockPromptTarget:
    fd, path = tempfile.mkstemp(suffix=".json.memory")
    file_memory = DuckDBMemory(db_path=":memory:")
    return MockPromptTarget(memory=file_memory)


def test_send_prompt_no_converter(mock_target: MockPromptTarget):
    orchestrator = PromptSendingOrchestrator(prompt_target=mock_target)

    orchestrator.send_text_prompts(["Hello"])
    assert mock_target.prompt_sent == ["Hello"]


@pytest.mark.asyncio
async def test_send_prompts_async_no_converter(mock_target: MockPromptTarget):
    orchestrator = PromptSendingOrchestrator(prompt_target=mock_target)

    await orchestrator.send_prompts_batch_async(["Hello"])
    assert mock_target.prompt_sent == ["Hello"]


def test_send_multiple_prompts_no_converter(mock_target: MockPromptTarget):
    orchestrator = PromptSendingOrchestrator(prompt_target=mock_target)

    orchestrator.send_text_prompts(["Hello", "my", "name"])
    assert mock_target.prompt_sent == ["Hello", "my", "name"]


@pytest.mark.asyncio
async def test_send_multiple_prompts_async_no_converter(mock_target: MockPromptTarget):
    orchestrator = PromptSendingOrchestrator(prompt_target=mock_target)

    await orchestrator.send_prompts_batch_async(["Hello", "my", "name"])
    assert mock_target.prompt_sent == ["Hello", "my", "name"]


def test_send_prompts_b64_converter(mock_target: MockPromptTarget):
    converter = Base64Converter()
    orchestrator = PromptSendingOrchestrator(prompt_target=mock_target, prompt_converters=[converter])

    orchestrator.send_text_prompts(["Hello"])
    assert mock_target.prompt_sent == ["SGVsbG8="]


def test_send_prompts_multiple_converters(mock_target: MockPromptTarget):
    b64_converter = Base64Converter()
    join_converter = StringJoinConverter(join_value="_")

    # This should base64 encode the prompt and then join the characters with an underscore
    converters = [b64_converter, join_converter]

    orchestrator = PromptSendingOrchestrator(prompt_target=mock_target, prompt_converters=converters)

    orchestrator.send_text_prompts(["Hello"])
    assert mock_target.prompt_sent == ["S_G_V_s_b_G_8_="]


def test_sendprompts_orchestrator_sets_target_memory(mock_target: MockPromptTarget):
    orchestrator = PromptSendingOrchestrator(prompt_target=mock_target)

    assert orchestrator._memory is mock_target._memory


def test_send_prompt_to_identifier(mock_target: MockPromptTarget):
    orchestrator = PromptSendingOrchestrator(prompt_target=mock_target)

    d = orchestrator.get_identifier()
    assert d["id"]
    assert d["__type__"] == "PromptSendingOrchestrator"
    assert d["__module__"] == "pyrit.orchestrator.prompt_sending_orchestrator"
