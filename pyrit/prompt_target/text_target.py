# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import asyncio
import csv
from pathlib import Path
import sys

from typing import IO

from pyrit.memory import MemoryInterface
from pyrit.models import PromptRequestResponse
from pyrit.models.prompt_request_piece import PromptRequestPiece
from pyrit.prompt_target import PromptTarget


class TextTarget(PromptTarget):
    """
    The TextTarget takes prompts, adds them to memory and writes them to io
    which is sys.stdout by default

    This can be useful in various situations, for example, if operators want to generate prompts
    but enter them manually.
    """

    def __init__(self, *, text_stream: IO[str] = sys.stdout, memory: MemoryInterface = None) -> None:
        super().__init__(memory=memory)
        self.stream_name = text_stream.name
        self._text_stream = text_stream

    def send_prompt(self, *, prompt_request: PromptRequestResponse) -> PromptRequestResponse:

        self._text_stream.write(f"{str(prompt_request)}\n")
        self._memory.add_request_response_to_memory(request=prompt_request)

        return None

    async def send_prompt_async(self, *, prompt_request: PromptRequestResponse) -> PromptRequestResponse:

        self._validate_request(prompt_request=prompt_request)
        await asyncio.sleep(0)

        return self.send_prompt(prompt_request=prompt_request)

    def import_scores_from_csv(self, csv_file_path: Path) -> list[PromptRequestPiece]:

        request_responses = []

        with open(csv_file_path, newline='') as csvfile:
            csvreader = csv.DictReader(csvfile)
            for row in csvreader:
                request_response = PromptRequestPiece(
                    role=row["role"],
                    original_value=row["value"],
                    original_value_data_type=row.get["data_type", None],
                    conversation_id=row.get("conversation_id", None),
                    sequence=row.get("sequence", None),
                    labels=row.get("labels", None),
                    response_error=row.get("response_error", None),
                    prompt_target_identifier=self.get_identifier(),
                )
                request_responses.append(request_response)

        # This is post validation, so the prompt_request_pieces should be okay and normalized
        self._memory.add_request_pieces_to_memory(request_responses=request_responses)
        return request_responses

    def _validate_request(self, *, prompt_request: PromptRequestResponse) -> None:
        pass
