# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

import hashlib

from pathlib import Path
from typing import Optional

from pydantic import BaseModel, ConfigDict


class PromptResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    # The text response for the prompt
    completion: str
    # The original prompt
    prompt: str = ""
    # An unique identifier for the response
    id: str = ""
    # The number of tokens used in the completion
    completion_tokens: int = 0
    # The number of tokens sent in the prompt
    prompt_tokens: int = 0
    # Total number of tokens used in the request
    total_tokens: int = 0
    # The model used
    model: str = ""
    # The type of operation (e.g., "text_completion")
    object: str = ""
    # When the object was created
    created_at: int = 0
    logprobs: Optional[bool] = False
    index: int = 0
    # Rationale why the model ended (e.g., "stop")
    finish_reason: str = ""
    # The time it took to complete the request from the moment the API request
    # was made, in nanoseconds.
    api_request_time_to_complete_ns: int = 0

    # Extra metadata that can be added to the response
    metadata: dict = {}

    def save_to_file(self, directory_path: Path) -> str:
        """Save the Prompt Response to disk and return the path of the new file.

        Args:
            directory_path: The path to save the file to
        Returns:
            The full path to the file that was saved
        """
        embedding_json = self.json()
        embedding_hash = hashlib.sha256(embedding_json.encode()).hexdigest()
        embedding_output_file_path = Path(directory_path, f"{embedding_hash}.json")
        embedding_output_file_path.write_text(embedding_json)
        return embedding_output_file_path.as_posix()

    def to_json(self) -> str:
        return self.model_dump_json()

    @staticmethod
    def load_from_file(file_path: Path) -> PromptResponse:
        """Load the Prompt Response from disk

        Args:
            file_path: The path to load the file from
        Returns:
            The loaded embedding response
        """
        embedding_json_data = file_path.read_text(encoding="utf-8")
        return PromptResponse.model_validate_json(embedding_json_data)
