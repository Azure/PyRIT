# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from pyrit.models import PromptRequestPiece


class PromptRequestResponse:
    def __init__(self, request_pieces: list[PromptRequestPiece]):
        self.request_pieces = request_pieces

    def __str__(self):
        ret = ""
        for request_piece in self.request_pieces:
            ret += str(request_piece) + "\n"
        return ret
