# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import tempfile
import pytest

from pyrit.memory import FileMemory
from pyrit.orchestrator import PromptSendingOrchestrator
from pyrit.prompt_target import PromptTarget
from pyrit.prompt_transformer import Base64Transformer


class MockPromptTarget(PromptTarget):
    count: int = 0

    def set_system_prompt(self, prompt: str, conversation_id: str, normalizer_id: str) -> None:
        self.system_prompt = prompt

    def send_prompt(self, normalized_prompt: str, conversation_id: str, normalizer_id: str) -> None:
        self.count += 1
        self.prompt = normalized_prompt


@pytest.fixture
def mock_target() -> MockPromptTarget:
    fd, path = tempfile.mkstemp(suffix=".json.memory")
    file_memory = FileMemory(filepath=path)
    return MockPromptTarget(memory=file_memory)


def test_send_prompt_no_transformer(mock_target: MockPromptTarget):
    orchestrator = PromptSendingOrchestrator(prompt_target=mock_target)

    orchestrator.send_prompts(["Hello"])
    assert mock_target.prompt == "Hello"


def test_send_multiple_prompts_no_transformer(mock_target: MockPromptTarget):
    orchestrator = PromptSendingOrchestrator(prompt_target=mock_target)

    orchestrator.send_prompts(["Hello", "my", "name"])
    assert mock_target.prompt == "name"
    assert mock_target.count == 3


def test_send_prompts_b64_transform(mock_target: MockPromptTarget):
    transformer = Base64Transformer()
    orchestrator = PromptSendingOrchestrator(prompt_target=mock_target, prompt_transformer=transformer)

    orchestrator.send_prompts(["Hello"])
    assert mock_target.prompt == "SGVsbG8="


def test_sendprompts_orchestrator_sets_target_memory(mock_target: MockPromptTarget):
    orchestrator = PromptSendingOrchestrator(prompt_target=mock_target)

    assert orchestrator.memory is mock_target.memory
