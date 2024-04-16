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

    def __str__(self):
        ret = ""
        for request_piece in self.request_pieces:
            ret += str(request_piece) + "\n"
        return "\n".join([str(request_piece) for request_piece in self.request_pieces])
