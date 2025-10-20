# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Optional, Sequence, get_args

from pyrit.models import Message, MessagePiece
from pyrit.models.literals import PromptDataType


class ScorerPromptValidator:
    def __init__(
        self,
        *,
        supported_data_types: Optional[Sequence[PromptDataType]] = None,
        required_metadata: Optional[Sequence[str]] = None,
        max_pieces_in_response: Optional[int] = None,
        enforce_all_pieces_valid: Optional[bool] = False,
        is_objective_required=False,
    ):
        if supported_data_types:
            self._supported_data_types = supported_data_types
        else:
            self._supported_data_types = get_args(PromptDataType)

        self._required_metadata = required_metadata or []

        self._max_pieces_in_response = max_pieces_in_response
        self._enforce_all_pieces_valid = enforce_all_pieces_valid

        self._is_objective_required = is_objective_required

    def validate(self, message: Message, objective: str | None) -> None:
        valid_pieces_count = 0
        for piece in message.message_pieces:
            if self.is_message_piece_supported(piece):
                valid_pieces_count += 1
            elif self._enforce_all_pieces_valid:
                raise ValueError(
                    f"Request piece {piece.id} with data type {piece.converted_value_data_type} is not supported."
                )

        if valid_pieces_count < 1:
            attempted_metadata = [getattr(piece, "prompt_metadata", None) for piece in message.message_pieces]
            raise ValueError(
                "There are no valid pieces to score. \n\n"
                f"Required types: {self._supported_data_types}. "
                f"Required metadata: {self._required_metadata}. "
                f"Length limit: {self._max_pieces_in_response}. "
                f"Objective required: {self._is_objective_required}. "
                f"Prompt pieces: {message.message_pieces}. "
                f"Prompt metadata: {attempted_metadata}. "
                f"Objective included: {objective}. "
            )

        if self._max_pieces_in_response is not None:
            if len(message.message_pieces) > self._max_pieces_in_response:
                raise ValueError(
                    f"Request response has {len(message.message_pieces)} pieces, "
                    f"exceeding the limit of {self._max_pieces_in_response}."
                )

        if self._is_objective_required and not objective:
            raise ValueError("Objective is required but not provided.")

    def is_message_piece_supported(self, message_piece: MessagePiece) -> bool:
        if message_piece.converted_value_data_type not in self._supported_data_types:
            return False

        for metadata in self._required_metadata:
            if metadata not in message_piece.prompt_metadata:
                return False

        return True
