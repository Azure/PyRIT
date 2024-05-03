# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import tempfile

from contextlib import AbstractAsyncContextManager
from typing import Generator, Optional
import uuid

from sqlalchemy import inspect

from pyrit.memory import DuckDBMemory, MemoryInterface
from pyrit.memory.memory_models import PromptMemoryEntry
from pyrit.models import PromptRequestResponse, PromptRequestPiece
from pyrit.orchestrator import Orchestrator
from pyrit.prompt_target.prompt_chat_target.prompt_chat_target import PromptChatTarget


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


class MockPromptTarget(PromptChatTarget):
    prompt_sent: list[str]

    def __init__(self, id=None, memory=None) -> None:
        self.id = id
        self.prompt_sent = []
        self._memory = memory

    def set_system_prompt(
        self,
        *,
        system_prompt: str,
        conversation_id: str,
        orchestrator_identifier: Optional[dict[str, str]] = None,
        labels: Optional[dict[str, str]] = None,
    ) -> None:
        self.system_prompt = system_prompt

    def send_prompt(self, *, prompt_request: PromptRequestResponse) -> PromptRequestResponse:
        self.prompt_sent.append(prompt_request.request_pieces[0].converted_value)
        return None

    async def send_prompt_async(self, *, prompt_request: PromptRequestResponse) -> PromptRequestResponse:
        self.prompt_sent.append(prompt_request.request_pieces[0].converted_value)
        return None

    def _validate_request(self, *, prompt_request: PromptRequestResponse) -> None:
        """
        Validates the provided prompt request response
        """
        pass


def get_memory_interface() -> Generator[MemoryInterface, None, None]:
    # Create an in-memory DuckDB engine
    duckdb_memory = DuckDBMemory(db_path=":memory:")

    duckdb_memory.disable_embedding()

    # Reset the database to ensure a clean state
    duckdb_memory.reset_database()
    inspector = inspect(duckdb_memory.engine)

    # Verify that tables are created as expected
    assert "PromptMemoryEntries" in inspector.get_table_names(), "PromptMemoryEntries table not created."
    assert "EmbeddingData" in inspector.get_table_names(), "EmbeddingData table not created."

    yield duckdb_memory
    duckdb_memory.dispose_engine()


def get_image_request_piece() -> PromptRequestPiece:
    file_name: str
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
        file_name = temp_file.name
        temp_file.write(b"image data")

        return PromptRequestPiece(
            role="user",
            original_value=file_name,
            converted_value=file_name,
            original_value_data_type="image_path",
            converted_value_data_type="image_path",
        )


def get_sample_conversations() -> list[PromptRequestPiece]:

    orchestrator1 = Orchestrator()
    orchestrator2 = Orchestrator()

    conversation_1 = str(uuid.uuid4())

    return [
        PromptRequestPiece(
            role="user",
            original_value="original prompt text",
            converted_value="Hello, how are you?",
            conversation_id=conversation_1,
            sequence=0,
            orchestrator_identifier=orchestrator1.get_identifier(),
        ),
        PromptRequestPiece(
            role="assistant",
            original_value="original prompt text",
            converted_value="I'm fine, thank you!",
            conversation_id=conversation_1,
            sequence=0,
            orchestrator_identifier=orchestrator1.get_identifier(),
        ),
        PromptRequestPiece(
            role="assistant",
            original_value="original prompt text",
            converted_value="I'm fine, thank you!",
            conversation_id=str(uuid.uuid4()),
            orchestrator_identifier=orchestrator2.get_identifier(),
        ),
    ]


def get_sample_conversation_entries() -> list[PromptMemoryEntry]:

    conversations = get_sample_conversations()
    return [PromptMemoryEntry(entry=conversation) for conversation in conversations]
