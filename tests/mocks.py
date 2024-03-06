# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from contextlib import AbstractAsyncContextManager

from pyrit.prompt_target import PromptTarget


class MockHttpPostAsync(AbstractAsyncContextManager):
    def __init__(self, url, headers=None, json=None, params=None, ssl=None):
        self.status = 200
        if url == "http://aml-test-endpoint.com":
            self._json = [{"0": "extracted response"}]
        else:
            raise NotImplementedError(f"No mock for HTTP POST {url}")

    async def json(self, content_type="application/json"):
        return self._json

    async def raise_for_status(self):
        if not (200 <= self.status < 300):
            raise Exception(f"HTTP Error {self.status}")

    async def __aexit__(self, exc_type, exc, tb):
        pass

    async def __aenter__(self):
        return self


class MockHttpPostSync:
    def __init__(self, url, headers=None, json=None, params=None, ssl=None):
        self.status = 200
        self.status_code = 200
        if url == "http://aml-test-endpoint.com":
            self._json = [{"0": "extracted response"}]
        else:
            raise NotImplementedError(f"No mock for HTTP POST {url}")

    def json(self, content_type="application/json"):
        return self._json

    def raise_for_status(self):
        if not (200 <= self.status < 300):
            raise Exception(f"HTTP Error {self.status}")


class MockPromptTarget(PromptTarget):
    prompt_sent: list[str]

    def __init__(self, id=None, memory=None) -> None:
        self.id = id
        self.prompt_sent = []
        self._memory = memory

    def set_system_prompt(self, prompt: str, conversation_id: str, normalizer_id: str) -> None:
        self.system_prompt = prompt

    def send_prompt(self, normalized_prompt: str, conversation_id: str, normalizer_id: str) -> None:
        self.prompt_sent.append(normalized_prompt)
