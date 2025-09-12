# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Optional, Sequence, get_args

from pyrit.models.literals import PromptDataType
from pyrit.models import PromptRequestPiece, PromptRequestResponse


class ScorerPromptValidator:
    def __init__(
        self,
        *,
        supported_data_types: Optional[Sequence[PromptDataType]] = None,
        required_metadata: Optional[Sequence[str]] = None,
        multi_part_response_length_limit: Optional[int] = None,
        enforce_all_pieces_valid: Optional[bool] = False,
        is_objective_required = False,
   ):
        if supported_data_types:
            self._supported_data_types = supported_data_types
        else:
            self._supported_data_types = get_args(PromptDataType)

        self._required_metadata = required_metadata or []

        self._multi_part_response_length_limit = multi_part_response_length_limit
        self._enforce_all_pieces_valid = enforce_all_pieces_valid

        self._is_objective_required = is_objective_required

    def validate(self, request_response: PromptRequestResponse, objective: str|None) -> None:
        valid_pieces_count = 0
        for piece in request_response.request_pieces:
            if self.is_request_piece_supported(piece):
                valid_pieces_count += 1
            elif self._enforce_all_pieces_valid:
                raise ValueError(
                    f"Request piece {piece.id} with data type {piece.converted_value_data_type} is not supported."
                )
        
        if valid_pieces_count < 1:
            attempted_metadata = [getattr(piece, 'prompt_metadata', None) for piece in request_response.request_pieces]
            raise ValueError(
                "There are no valid pieces to score. \n\n"
                f"Requireed types: {self._supported_data_types}. "
                f"Required metadata: {self._required_metadata}. "
                f"Length limit: {self._multi_part_response_length_limit}. "
                f"Objective required: {self._is_objective_required}. "
                f"Prompt pieces: {request_response.request_pieces}. "
                f"Prompt metadata: {attempted_metadata}. "
                f"Objective included: {objective}. "
            )
                
        if self._multi_part_response_length_limit is not None:
            if len(request_response.request_pieces) > self._multi_part_response_length_limit:
                raise ValueError(
                    f"Request response has {len(request_response.request_pieces)} pieces, "
                    f"exceeding the limit of {self._multi_part_response_length_limit}."
                )
            
        if self._is_objective_required and not objective:
            raise ValueError("Objective is required but not provided.")

    def is_request_piece_supported(self, request_piece: PromptRequestPiece) -> bool:
        if request_piece.converted_value_data_type not in self._supported_data_types:
            return False
        
        for metadata in self._required_metadata:
            if metadata not in request_piece.prompt_metadata:
                return False

        return True