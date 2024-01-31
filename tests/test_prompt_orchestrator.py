# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import pathlib
import tempfile
from unittest.mock import patch

import pytest
from openai.types.chat import ChatCompletion, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice

from pyrit.agent import RedTeamingBot
from pyrit.chat import AzureOpenAIChat
from pyrit.models import PromptTemplate, Score
from pyrit.memory import FileMemory
from pyrit.common.path import HOME_PATH
from pyrit.orchestrator.send_all_prompts_orchestrator import SendAllPromptsOrchestrator
from pyrit.prompt_target.azure_openai_chat_target import AzureOpenAIChatTarget
from pyrit.prompt_target import PromptTarget
from pyrit.prompt_transformer.base64_transformer import Base64Transformer


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
    file_memory = FileMemory(filepath= path)
    return MockPromptTarget(memory=file_memory)

def test_send_prompt_no_transformer(mock_target: MockPromptTarget):
    orchestrator = SendAllPromptsOrchestrator(prompt_target=mock_target)

    orchestrator.send_prompts(["Hello"])
    assert mock_target.prompt == "Hello"

def test_send_multiple_prompts_no_transformer(mock_target: MockPromptTarget):
    orchestrator = SendAllPromptsOrchestrator(prompt_target=mock_target)

    orchestrator.send_prompts(["Hello", "my", "name"])
    assert mock_target.prompt == "name"
    assert mock_target.count == 3


def test_send_prompts_b64_transform(mock_target: MockPromptTarget):
    transformer = Base64Transformer()
    orchestrator = SendAllPromptsOrchestrator(prompt_target=mock_target, prompt_transformer=transformer)

    orchestrator.send_prompts(["Hello"])
    assert mock_target.prompt == 'SGVsbG8='