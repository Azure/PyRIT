# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Dict, MutableSequence, Optional, Sequence, Union

from pyrit.common.utils import combine_dict
from pyrit.models.literals import ChatMessageRole, PromptDataType, PromptResponseError
from pyrit.models.prompt_request_piece import PromptRequestPiece


class PromptRequestResponse:
    """
    Represents a response to a prompt request.

    This is a single request to a target. It can contain multiple prompt request pieces.

    Parameters:
        request_pieces (Sequence[PromptRequestPiece]): The list of prompt request pieces.
    """

    def __init__(self, request_pieces: Sequence[PromptRequestPiece]):
        self.request_pieces = request_pieces

    def get_value(self, n: int = 0) -> str:
        """Return the converted value of the nth request piece."""
        if n >= len(self.request_pieces):
            raise IndexError(f"No request piece at index {n}.")
        return self.request_pieces[n].converted_value

    def get_values(self) -> list[str]:
        """Return the converted values of all request pieces."""
        return [request_piece.converted_value for request_piece in self.request_pieces]

    def validate(self):
        """
        Validates the request response.
        """
        if len(self.request_pieces) == 0:
            raise ValueError("Empty request pieces.")

        conversation_id = self.request_pieces[0].conversation_id
        role = None
        for request_piece in self.request_pieces:

            if request_piece.conversation_id != conversation_id:
                raise ValueError("Conversation ID mismatch.")

            if not request_piece.converted_value:
                raise ValueError("Converted prompt text is None.")

            if not role:
                role = request_piece.role

            elif role != request_piece.role:
                raise ValueError("Inconsistent roles within the same prompt request response entry.")

    def __str__(self):
        ret = ""
        for request_piece in self.request_pieces:
            ret += str(request_piece) + "\n"
        return "\n".join([str(request_piece) for request_piece in self.request_pieces])

    @staticmethod
    def flatten_to_prompt_request_pieces(
        request_responses: Sequence["PromptRequestResponse"],
    ) -> MutableSequence[PromptRequestPiece]:
        if not request_responses:
            return []
        response_pieces: MutableSequence[PromptRequestPiece] = []

        for response in request_responses:
            response_pieces.extend(response.request_pieces)

        return response_pieces

    @classmethod
    def from_prompt(cls, *, prompt: str, role: ChatMessageRole) -> "PromptRequestResponse":
        piece = PromptRequestPiece(original_value=prompt, role=role)
        return cls(request_pieces=[piece])

    @classmethod
    def from_system_prompt(cls, system_prompt: str) -> "PromptRequestResponse":
        return cls.from_prompt(prompt=system_prompt, role="system")


def group_conversation_request_pieces_by_sequence(
    request_pieces: Sequence[PromptRequestPiece],
) -> MutableSequence[PromptRequestResponse]:
    """
    Groups prompt request pieces from the same conversation into PromptRequestResponses.

    This is done using the sequence number and conversation ID.

    Args:
        request_pieces (Sequence[PromptRequestPiece]): A list of PromptRequestPiece objects representing individual
            request pieces.

    Returns:
        MutableSequence[PromptRequestResponse]: A list of PromptRequestResponse objects representing grouped request
            pieces. This is ordered by the sequence number

    Raises:
        ValueError: If the conversation ID of any request piece does not match the conversation ID of the first
        request piece.

    Example:
    >>> request_pieces = [
    >>>     PromptRequestPiece(conversation_id=1, sequence=1, text="Hello"),
    >>>     PromptRequestPiece(conversation_id=1, sequence=2, text="How are you?"),
    >>>     PromptRequestPiece(conversation_id=1, sequence=1, text="Hi"),
    >>>     PromptRequestPiece(conversation_id=1, sequence=2, text="I'm good, thanks!")
    >>> ]
    >>> grouped_responses = group_conversation_request_pieces(request_pieces)
    ... [
    ...     PromptRequestResponse(request_pieces=[
    ...         PromptRequestPiece(conversation_id=1, sequence=1, text="Hello"),
    ...         PromptRequestPiece(conversation_id=1, sequence=1, text="Hi")
    ...     ]),
    ...     PromptRequestResponse(request_pieces=[
    ...         PromptRequestPiece(conversation_id=1, sequence=2, text="How are you?"),
    ...         PromptRequestPiece(conversation_id=1, sequence=2, text="I'm good, thanks!")
    ...     ])
    ... ]
    """

    if not request_pieces:
        return []

    conversation_id = request_pieces[0].conversation_id

    conversation_by_sequence: dict[int, list[PromptRequestPiece]] = {}

    for request_piece in request_pieces:
        if request_piece.conversation_id != conversation_id:
            raise ValueError("Conversation ID must match.")

        if request_piece.sequence not in conversation_by_sequence:
            conversation_by_sequence[request_piece.sequence] = [request_piece]
        else:
            conversation_by_sequence[request_piece.sequence].append(request_piece)

    sorted_sequences = sorted(conversation_by_sequence.keys())
    return [PromptRequestResponse(conversation_by_sequence[seq]) for seq in sorted_sequences]


def construct_response_from_request(
    request: PromptRequestPiece,
    response_text_pieces: list[str],
    response_type: PromptDataType = "text",
    prompt_metadata: Optional[Dict[str, Union[str, int]]] = None,
    error: PromptResponseError = "none",
) -> PromptRequestResponse:
    """
    Constructs a response entry from a request.
    """

    # prompt metadat should be combined with the request piece metadata
    if request.prompt_metadata:
        prompt_metadata = combine_dict(request.prompt_metadata, prompt_metadata)

    return PromptRequestResponse(
        request_pieces=[
            PromptRequestPiece(
                role="assistant",
                original_value=resp_text,
                conversation_id=request.conversation_id,
                labels=request.labels,
                prompt_target_identifier=request.prompt_target_identifier,
                orchestrator_identifier=request.orchestrator_identifier,
                original_value_data_type=response_type,
                converted_value_data_type=response_type,
                prompt_metadata=prompt_metadata,
                response_error=error,
            )
            for resp_text in response_text_pieces
        ]
    )
