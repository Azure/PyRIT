# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from pyrit.models import PromptRequestPiece


class PromptRequestResponse:
    """
    Represents a response to a prompt request.

    This is a single request to a target. It can contain multiple prompt request pieces.

    Attributes:
        request_pieces (list[PromptRequestPiece]): The list of prompt request pieces.
    """

    def __init__(self, request_pieces: list[PromptRequestPiece]):
        self.request_pieces = request_pieces

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


def group_conversation_request_pieces_by_sequence(
    request_pieces: list[PromptRequestPiece],
) -> list[PromptRequestResponse]:
    """
    Groups prompt request pieces from the same conversation into PromptRequestResponses.

    This is done using the sequence number and conversation ID.

    Args:
        request_pieces (list[PromptRequestPiece]): A list of PromptRequestPiece objects representing individual
            request pieces.

    Returns:
        list[PromptRequestResponse]: A list of PromptRequestResponse objects representing grouped request pieces.
            this is ordered by the sequence number

    Raises:
        ValueError: If the conversation ID of any request piece does not match the conversation ID of the first
        request piece.

    Example:
        request_pieces = [
            PromptRequestPiece(conversation_id=1, sequence=1, text="Hello"),
            PromptRequestPiece(conversation_id=1, sequence=2, text="How are you?"),
            PromptRequestPiece(conversation_id=1, sequence=1, text="Hi"),
            PromptRequestPiece(conversation_id=1, sequence=2, text="I'm good, thanks!")
        ]

        grouped_responses = group_conversation_request_pieces(request_pieces) ->

        Returns:
         [
             PromptRequestResponse(request_pieces=[
                 PromptRequestPiece(conversation_id=1, sequence=1, text="Hello"),
                 PromptRequestPiece(conversation_id=1, sequence=1, text="Hi")
             ]),
             PromptRequestResponse(request_pieces=[
                 PromptRequestPiece(conversation_id=1, sequence=2, text="How are you?"),
                 PromptRequestPiece(conversation_id=1, sequence=2, text="I'm good, thanks!")
             ])
         ]
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
