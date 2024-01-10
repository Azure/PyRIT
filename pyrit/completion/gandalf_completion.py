# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import enum

import requests

from pyrit.interfaces import CompletionSupport
from pyrit.models import PromptResponse


class GandalfLevel(enum.Enum):
    LEVEL_1 = "baseline"
    LEVEL_2 = "do-not-tell"
    LEVEL_3 = "do-not-tell-and-block"
    LEVEL_4 = "gpt-is-password-encoded"
    LEVEL_5 = "word-blacklist"
    LEVEL_6 = "gpt-blacklist"
    LEVEL_7 = "gandalf"
    LEVEL_8 = "gandalf-the-white"
    LEVEL_9 = "adventure-1"
    LEVEL_10 = "adventure-2"


class GandalfCompletionEngine(CompletionSupport):
    _endpoint: str
    _defender: str

    def __init__(self, level: GandalfLevel):
        self._endpoint = "https://gandalf.lakera.ai/api/send-message"
        self._defender = level.value

    def complete_text(self, text: str, **kwargs) -> PromptResponse:
        payload = {
            "defender": self._defender,
            "prompt": text,
        }
        response = requests.post(self._endpoint, data=payload)
        if response.status_code == 200:
            return PromptResponse(completion=response.json()["answer"])
        else:
            raise Exception(
                f"Error in Gandalf Completion Engine. "
                f"Status code returned {response.status_code}, message: {response.text}"
            )

    async def complete_text_async(self, text: str, **kwargs) -> PromptResponse:
        raise NotImplementedError
