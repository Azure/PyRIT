# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import abc
import uuid

from datetime import datetime
from typing import Dict, List, Optional, Literal, get_args
from uuid import uuid4

from pyrit.models import ChatMessage, data_serializer_factory, ChatMessageRole, PromptDataType, PromptResponseError


Originator = Literal["orchestrator", "converter", "undefined", "scorer"]


class PromptRequestPiece(abc.ABC):
    """
    Represents a prompt request piece.

    Attributes:
        id (UUID): The unique identifier for the memory entry.
        role (PromptType): system, assistant, user
        conversation_id (str): The identifier for the conversation which is associated with a single target.
        sequence (int): The order of the conversation within a conversation_id.
            Can be the same number for multi-part requests or multi-part responses.
        timestamp (DateTime): The timestamp of the memory entry.
        labels (Dict[str, str]): The labels associated with the memory entry. Several can be standardized.
        prompt_metadata (JSON): The metadata associated with the prompt. This can be specific to any scenarios.
            Because memory is how components talk with each other, this can be component specific.
            e.g. the URI from a file uploaded to a blob store, or a document type you want to upload.
        converters (list[PromptConverter]): The converters for the prompt.
        prompt_target (PromptTarget): The target for the prompt.
        orchestrator_identifier (Dict[str, str]): The orchestrator identifier for the prompt.
        original_value_data_type (PromptDataType): The data type of the original prompt (text, image)
        original_value (str): The text of the original prompt. If prompt is an image, it's a link.
        original_value_sha256 (str): The SHA256 hash of the original prompt data.
        converted_value_data_type (PromptDataType): The data type of the converted prompt (text, image)
        converted_value (str): The text of the converted prompt. If prompt is an image, it's a link.
        converted_value_sha256 (str): The SHA256 hash of the original prompt data.

    Methods:
        __str__(): Returns a string representation of the memory entry.
    """

    def __init__(
        self,
        *,
        role: ChatMessageRole,
        original_value: str,
        converted_value: Optional[str] = None,
        id: Optional[uuid.UUID] = None,
        conversation_id: Optional[str] = None,
        sequence: int = -1,
        labels: Optional[Dict[str, str]] = None,
        prompt_metadata: Optional[str] = None,
        converter_identifiers: Optional[List[Dict[str, str]]] = None,
        prompt_target_identifier: Optional[Dict[str, str]] = None,
        orchestrator_identifier: Optional[Dict[str, str]] = None,
        scorer_identifier: Dict[str, str] = None,
        original_value_data_type: PromptDataType = "text",
        converted_value_data_type: PromptDataType = "text",
        response_error: PromptResponseError = "none",
        originator: Originator = "undefined",
    ):

        self.id = id if id else uuid4()

        if role not in ChatMessageRole.__args__:  # type: ignore
            raise ValueError(f"Role {role} is not a valid role.")

        self.role = role

        if converted_value is None:
            converted_value = original_value

        self.conversation_id = conversation_id if conversation_id else str(uuid4())
        self.sequence = sequence

        self.timestamp = datetime.utcnow()
        self.labels = labels
        self.prompt_metadata = prompt_metadata

        self.converter_identifiers = converter_identifiers

        self.prompt_target_identifier = prompt_target_identifier
        self.orchestrator_identifier = orchestrator_identifier
        self.scorer_identifier = scorer_identifier

        self._original_value = original_value

        if original_value_data_type not in get_args(PromptDataType):
            raise ValueError(f"original_value_data_type {original_value_data_type} is not a valid data type.")

        self.original_value_data_type = original_value_data_type

        original_serializer = data_serializer_factory(data_type=original_value_data_type, value=original_value)
        self._original_value_sha256 = original_serializer.get_sha256()

        self._converted_value = converted_value

        if converted_value_data_type not in get_args(PromptDataType):
            raise ValueError(f"converted_value_data_type {converted_value_data_type} is not a valid data type.")

        self.converted_value_data_type = converted_value_data_type

        converted_serializer = data_serializer_factory(data_type=converted_value_data_type, value=converted_value)
        self._converted_value_sha256 = converted_serializer.get_sha256()

        if response_error not in get_args(PromptResponseError):
            raise ValueError(f"response_error {response_error} is not a valid response error.")

        self.response_error = response_error
        self.originator = originator

    @property
    def converted_value(self) -> str:
        return self._converted_value

    @converted_value.setter
    def converted_value(self, value: str):
        self._converted_value = value
        converted_serializer = data_serializer_factory(data_type=self.converted_value_data_type, value=value)
        self._converted_value_sha256 = converted_serializer.get_sha256()

    @property
    def converted_value_sha256(self):
        return self._converted_value_sha256

    @property
    def original_value(self) -> str:
        return self._original_value

    @original_value.setter
    def original_value(self, value: str):
        self._original_value = value
        original_serializer = data_serializer_factory(data_type=self.original_value_data_type, value=value)
        self._original_value_sha256 = original_serializer.get_sha256()

    @property
    def original_value_sha256(self):
        return self._original_value_sha256

    def to_chat_message(self) -> ChatMessage:
        return ChatMessage(role=self.role, content=self.converted_value)

    def to_prompt_request_response(self) -> "PromptRequestResponse":  # type: ignore # noqa F821
        from pyrit.models.prompt_request_response import PromptRequestResponse

        return PromptRequestResponse([self])  # noqa F821

    def __str__(self):
        return f"{self.prompt_target_identifier}: {self.role}: {self.converted_value}"
