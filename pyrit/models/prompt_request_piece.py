# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import abc
import hashlib
import uuid

from datetime import datetime
from typing import Dict, Literal, List
from uuid import uuid4

from pyrit.models import ChatMessage, ChatMessageRole


PromptDataType = Literal["text", "image_path", "url"]

"""
The type of the error in the prompt response
blocked: blocked by an external filter e.g. Azure Filters
model: the model refused to answer or request e.g. "I'm sorry..."
processing: there is an exception thrown unrelated to the query
unknown: the type of error is unknown
"""
PromptResponseError = Literal["none", "blocked", "model", "processing", "unknown"]


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
        orchestrator (Orchestrator): The orchestrator for the prompt.
        original_prompt_data_type (PromptDataType): The data type of the original prompt (text, image)
        original_prompt_text (str): The text of the original prompt. If prompt is an image, it's a link.
        original_prompt_data_sha256 (str): The SHA256 hash of the original prompt data.
        converted_prompt_data_type (PromptDataType): The data type of the converted prompt (text, image)
        converted_prompt_text (str): The text of the converted prompt. If prompt is an image, it's a link.
        converted_prompt_data_sha256 (str): The SHA256 hash of the original prompt data.

    Methods:
        __str__(): Returns a string representation of the memory entry.
    """

    def __init__(
        self,
        *,
        role: ChatMessageRole,
        original_prompt_text: str,
        converted_prompt_text: str = None,
        id: uuid.UUID = None,
        conversation_id: str = None,
        sequence: int = -1,
        labels: Dict[str, str] = None,
        prompt_metadata: str = None,
        converter_identifiers: List[Dict[str, str]] = None,
        prompt_target_identifier: Dict[str, str] = None,
        orchestrator_identifier: Dict[str, str] = None,
        original_prompt_data_type: PromptDataType = "text",
        converted_prompt_data_type: PromptDataType = "text",
        response_error: PromptResponseError = "none",
    ):

        self.id = id if id else uuid4()

        self.role = role

        if converted_prompt_text is None:
            converted_prompt_text = original_prompt_text

        self.conversation_id = conversation_id if conversation_id else str(uuid4())
        self.sequence = sequence

        self.timestamp = datetime.utcnow()
        self.labels = labels
        self.prompt_metadata = prompt_metadata

        self.converter_identifiers = converter_identifiers

        self.prompt_target_identifier = prompt_target_identifier
        self.orchestrator_identifier = orchestrator_identifier

        self.original_prompt_text = original_prompt_text
        self.original_prompt_data_type = original_prompt_data_type
        self.original_prompt_data_sha256 = self._create_sha256(original_prompt_text)

        self.converted_prompt_data_type = converted_prompt_data_type
        self.converted_prompt_text = converted_prompt_text
        self.converted_prompt_data_sha256 = self._create_sha256(converted_prompt_text)

        self.response_error = response_error

    def is_sequence_set(self) -> bool:
        return self.sequence != -1

    def to_chat_message(self) -> ChatMessage:
        return ChatMessage(role=self.role, content=self.converted_prompt_text)

    def to_prompt_request_response(self) -> "PromptRequestResponse":  # type: ignore # noqa F821
        from pyrit.models.prompt_request_response import PromptRequestResponse

        return PromptRequestResponse([self])  # noqa F821

    def _create_sha256(self, text: str) -> str:
        input_bytes = text.encode("utf-8")
        hash_object = hashlib.sha256(input_bytes)
        return hash_object.hexdigest()

    def __str__(self):
        return f"{self.prompt_target_identifier}: {self.role}: {self.converted_prompt_text}"
