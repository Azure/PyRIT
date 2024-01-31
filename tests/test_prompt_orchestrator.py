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


class MockPromptTarget(PromptTarget):
    def set_system_prompt(self, prompt: str, conversation_id: str, normalizer_id: str) -> None:
        self.system_prompt = prompt

    def send_prompt(self, normalized_prompt: str, conversation_id: str, normalizer_id: str) -> None:
        self.prompt = normalized_prompt





def test_send_prompts():
    fd, path = tempfile.mkstemp(suffix=".json.memory")
    file_memory = FileMemory(filepath= path)

    mockTarget = MockPromptTarget(memory=file_memory)
    orchestrator = SendAllPromptsOrchestrator(prompt_target=mockTarget)

    orchestrator.send_prompts(["Hello"])
    assert mockTarget.prompt == "Hello"
