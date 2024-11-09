# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import csv
import json
from pathlib import Path
import sys

from typing import IO

from pyrit.models import PromptRequestResponse, PromptRequestPiece
from pyrit.prompt_target import PromptTarget


class TextTarget(PromptTarget):
    """
    The TextTarget takes prompts, adds them to memory and writes them to io
    which is sys.stdout by default

    This can be useful in various situations, for example, if operators want to generate prompts
    but enter them manually.
    """

    def __init__(
        self,
        *,
        text_stream: IO[str] = sys.stdout,
    ) -> None:
        super().__init__()
        self._text_stream = text_stream

    async def send_prompt_async(self, *, prompt_request: PromptRequestResponse) -> PromptRequestResponse:

        self._validate_request(prompt_request=prompt_request)

        self._text_stream.write(f"{str(prompt_request)}\n")
        self._text_stream.flush()

        return None

    def import_scores_from_csv(self, csv_file_path: Path) -> list[PromptRequestPiece]:

        request_responses = []

        with open(csv_file_path, newline="") as csvfile:
            csvreader = csv.DictReader(csvfile)

            for row in csvreader:
                sequence_str = row.get("sequence", None)
                labels_str = row.get("labels", None)
                labels = json.loads(labels_str) if labels_str else None

                request_response = PromptRequestPiece(
                    role=row["role"],  # type: ignore
                    original_value=row["value"],
                    original_value_data_type=row.get["data_type", None],  # type: ignore
                    conversation_id=row.get("conversation_id", None),
                    sequence=int(sequence_str) if sequence_str else None,
                    labels=labels,
                    response_error=row.get("response_error", None),  # type: ignore
                    prompt_target_identifier=self.get_identifier(),
                )
                request_responses.append(request_response)

        # This is post validation, so the prompt_request_pieces should be okay and normalized
        self._memory.add_request_pieces_to_memory(request_pieces=request_responses)
        return request_responses

    def _validate_request(self, *, prompt_request: PromptRequestResponse) -> None:
        pass
