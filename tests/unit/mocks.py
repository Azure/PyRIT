# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import tempfile
import uuid
from contextlib import AbstractAsyncContextManager
from typing import Generator, MutableSequence, Optional, Sequence
from unittest.mock import MagicMock, patch

from mock_alchemy.mocking import UnifiedAlchemyMagicMock

from pyrit.memory import AzureSQLMemory, CentralMemory, PromptMemoryEntry
from pyrit.models import PromptRequestPiece, PromptRequestResponse
from pyrit.orchestrator import Orchestrator
from pyrit.prompt_target import PromptChatTarget, limit_requests_per_minute


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

    def __init__(self, id=None, rpm=None) -> None:
        super().__init__(max_requests_per_minute=rpm)
        self.id = id
        self.prompt_sent = []

    def set_system_prompt(
        self,
        *,
        system_prompt: str,
        conversation_id: str,
        orchestrator_identifier: Optional[dict[str, str]] = None,
        labels: Optional[dict[str, str]] = None,
    ) -> None:
        self.system_prompt = system_prompt
        if self._memory:
            self._memory.add_request_response_to_memory(
                request=PromptRequestPiece(
                    role="system",
                    original_value=system_prompt,
                    converted_value=system_prompt,
                    conversation_id=conversation_id,
                    orchestrator_identifier=orchestrator_identifier,
                    labels=labels,
                ).to_prompt_request_response()
            )

    @limit_requests_per_minute
    async def send_prompt_async(self, *, prompt_request: PromptRequestResponse) -> PromptRequestResponse:
        self.prompt_sent.append(prompt_request.get_value())

        return PromptRequestPiece(
            role="assistant",
            original_value="default",
            conversation_id=prompt_request.request_pieces[0].conversation_id,
            orchestrator_identifier=prompt_request.request_pieces[0].orchestrator_identifier,
            labels=prompt_request.request_pieces[0].labels,
        ).to_prompt_request_response()

    def _validate_request(self, *, prompt_request: PromptRequestResponse) -> None:
        """
        Validates the provided prompt request response
        """

    def is_json_response_supported(self) -> bool:
        return False


def get_azure_sql_memory() -> Generator[AzureSQLMemory, None, None]:
    # Create a test Azure SQL Server DB
    with (
        patch("pyrit.memory.AzureSQLMemory.get_session") as get_session_mock,
        patch("pyrit.memory.AzureSQLMemory._create_auth_token") as create_auth_token_mock,
        patch("pyrit.memory.AzureSQLMemory._enable_azure_authorization") as enable_azure_authorization_mock,
    ):
        os.environ[AzureSQLMemory.AZURE_STORAGE_ACCOUNT_DB_DATA_CONTAINER_URL] = (
            "https://test.blob.core.windows.net/test"
        )
        os.environ[AzureSQLMemory.AZURE_STORAGE_ACCOUNT_DB_DATA_SAS_TOKEN] = "valid_sas_token"

        azure_sql_memory = AzureSQLMemory(
            connection_string="mssql+pyodbc://test:test@test/test?driver=ODBC+Driver+18+for+SQL+Server",
            results_container_url=os.environ[AzureSQLMemory.AZURE_STORAGE_ACCOUNT_DB_DATA_CONTAINER_URL],
            results_sas_token=os.environ[AzureSQLMemory.AZURE_STORAGE_ACCOUNT_DB_DATA_SAS_TOKEN],
        )

        session_mock = UnifiedAlchemyMagicMock()
        session_mock.__enter__.return_value = session_mock
        session_mock.is_modified.return_value = True
        get_session_mock.return_value = session_mock

        create_auth_token_mock.return_value = "token"
        enable_azure_authorization_mock.return_value = None

        azure_sql_memory.disable_embedding()
        CentralMemory.set_memory_instance(azure_sql_memory)
        yield azure_sql_memory

    azure_sql_memory.dispose_engine()


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


def get_audio_request_piece() -> PromptRequestPiece:
    file_name: str
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp_file:
        file_name = temp_file.name
        temp_file.write(b"audio data")

        return PromptRequestPiece(
            role="user",
            original_value=file_name,
            converted_value=file_name,
            original_value_data_type="audio_path",
            converted_value_data_type="audio_path",
        )


def get_test_request_piece() -> PromptRequestPiece:

    return PromptRequestPiece(
        role="user",
        original_value="some text",
        converted_value="some text",
        original_value_data_type="text",
        converted_value_data_type="text",
    )


def get_sample_conversations() -> MutableSequence[PromptRequestPiece]:
    with patch.object(CentralMemory, "get_memory_instance", return_value=MagicMock()):
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


def get_sample_conversation_entries() -> Sequence[PromptMemoryEntry]:
    conversations = get_sample_conversations()
    return [PromptMemoryEntry(entry=conversation) for conversation in conversations]


def openai_response_json_dict() -> dict:
    return {
        "id": "12345678-1a2b-3c4e5f-a123-12345678abcd",
        "object": "chat.completion",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "hi"},
                "finish_reason": "stop",
            }
        ],
        "model": "gpt-4-v",
    }
