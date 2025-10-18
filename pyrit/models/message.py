# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Dict, MutableSequence, Optional, Sequence, Union

from pyrit.common.utils import combine_dict
from pyrit.models.literals import ChatMessageRole, PromptDataType, PromptResponseError
from pyrit.models.message_piece import MessagePiece


class Message:
    """
    Represents a response to a prompt request.

    This is a single request to a target. It can contain multiple prompt request pieces.

    Parameters:
        request_pieces (Sequence[MessagePiece]): The list of prompt request pieces.
    """

    def __init__(self, request_pieces: Sequence[MessagePiece], *, skip_validation: Optional[bool] = False):
        if not request_pieces:
            raise ValueError("Message must have at least one request piece.")
        self.request_pieces = request_pieces
        if not skip_validation:
            self.validate()

    def get_value(self, n: int = 0) -> str:
        """Return the converted value of the nth request piece."""
        if n >= len(self.request_pieces):
            raise IndexError(f"No request piece at index {n}.")
        return self.request_pieces[n].converted_value

    def get_values(self) -> list[str]:
        """Return the converted values of all request pieces."""
        return [request_piece.converted_value for request_piece in self.request_pieces]

    def get_piece(self, n: int = 0) -> MessagePiece:
        """Return the nth request piece."""
        if len(self.request_pieces) == 0:
            raise ValueError("Empty request pieces.")

        if n >= len(self.request_pieces):
            raise IndexError(f"No request piece at index {n}.")

        return self.request_pieces[n]

    def get_role(self) -> ChatMessageRole:
        """Return the role of the first request."""
        if len(self.request_pieces) == 0:
            raise ValueError("Empty request pieces.")

        return self.request_pieces[0].role

    def is_error(self) -> bool:
        """
        Returns True if any of the request pieces has an error response.
        """
        for piece in self.request_pieces:
            if piece.response_error != "none" or piece.converted_value_data_type == "error":
                return True
        return False

    def set_response_not_in_database(self):
        """
        Set that the prompt is not in the database.

        This is needed when we're scoring prompts or other things that have not been sent by PyRIT
        """
        for piece in self.request_pieces:
            piece.set_piece_not_in_database()

    def validate(self):
        """
        Validates the request response.
        """
        if len(self.request_pieces) == 0:
            raise ValueError("Empty request pieces.")

        conversation_id = self.request_pieces[0].conversation_id
        sequence = self.request_pieces[0].sequence
        role = self.request_pieces[0].role
        for request_piece in self.request_pieces:

            if request_piece.conversation_id != conversation_id:
                raise ValueError("Conversation ID mismatch.")

            if request_piece.sequence != sequence:
                raise ValueError("Inconsistent sequences within the same prompt request response entry.")

            if request_piece.converted_value is None:
                raise ValueError("Converted prompt text is None.")

            if request_piece.role != role:
                raise ValueError("Inconsistent roles within the same prompt request response entry.")

    def __str__(self):
        ret = ""
        for request_piece in self.request_pieces:
            ret += str(request_piece) + "\n"
        return "\n".join([str(request_piece) for request_piece in self.request_pieces])

    @staticmethod
    def get_all_values(request_responses: Sequence["Message"]) -> list[str]:
        """Return all converted values across the provided request responses."""
        values: list[str] = []
        for request_response in request_responses:
            values.extend(request_response.get_values())
        return values

    @staticmethod
    def flatten_to_prompt_request_pieces(
        request_responses: Sequence["Message"],
    ) -> MutableSequence[MessagePiece]:
        if not request_responses:
            return []
        response_pieces: MutableSequence[MessagePiece] = []

        for response in request_responses:
            response_pieces.extend(response.request_pieces)

        return response_pieces

    @classmethod
    def from_prompt(cls, *, prompt: str, role: ChatMessageRole) -> "Message":
        piece = MessagePiece(original_value=prompt, role=role)
        return cls(request_pieces=[piece])

    @classmethod
    def from_system_prompt(cls, system_prompt: str) -> "Message":
        return cls.from_prompt(prompt=system_prompt, role="system")


def group_conversation_request_pieces_by_sequence(
    request_pieces: Sequence[MessagePiece],
) -> MutableSequence[Message]:
    """
    Groups prompt request pieces from the same conversation into Messages.

    This is done using the sequence number and conversation ID.

    Args:
        request_pieces (Sequence[MessagePiece]): A list of MessagePiece objects representing individual
            request pieces.

    Returns:
        MutableSequence[Message]: A list of Message objects representing grouped request
            pieces. This is ordered by the sequence number

    Raises:
        ValueError: If the conversation ID of any request piece does not match the conversation ID of the first
        request piece.

    Example:
    >>> request_pieces = [
    >>>     MessagePiece(conversation_id=1, sequence=1, text="Given this list of creatures, which is your
    >>>     favorite:"),
    >>>     MessagePiece(conversation_id=1, sequence=2, text="Good question!"),
    >>>     MessagePiece(conversation_id=1, sequence=1, text="Raccoon, Narwhal, or Sloth?"),
    >>>     MessagePiece(conversation_id=1, sequence=2, text="I'd have to say raccoons are my favorite!"),
    >>> ]
    >>> grouped_responses = group_conversation_request_pieces(request_pieces)
    ... [
    ...     Message(request_pieces=[
    ...         MessagePiece(conversation_id=1, sequence=1, text="Given this list of creatures, which is your
    ...         favorite:"),
    ...         MessagePiece(conversation_id=1, sequence=1, text="Raccoon, Narwhal, or Sloth?")
    ...     ]),
    ...     Message(request_pieces=[
    ...         MessagePiece(conversation_id=1, sequence=2, text="Good question!"),
    ...         MessagePiece(conversation_id=1, sequence=2, text="I'd have to say raccoons are my favorite!")
    ...     ])
    ... ]
    """

    if not request_pieces:
        return []

    conversation_id = request_pieces[0].conversation_id

    conversation_by_sequence: dict[int, list[MessagePiece]] = {}

    for request_piece in request_pieces:
        if request_piece.conversation_id != conversation_id:
            raise ValueError(
                f"All request pieces must be from the same conversation. "
                f"Expected conversation_id='{conversation_id}', but found '{request_piece.conversation_id}'. "
                f"If grouping pieces from multiple conversations, group by conversation_id first."
            )

        if request_piece.sequence not in conversation_by_sequence:
            conversation_by_sequence[request_piece.sequence] = []
        conversation_by_sequence[request_piece.sequence].append(request_piece)

    sorted_sequences = sorted(conversation_by_sequence.keys())
    return [Message(conversation_by_sequence[seq]) for seq in sorted_sequences]


def group_request_pieces_into_conversations(
    request_pieces: Sequence[MessagePiece],
) -> list[list[Message]]:
    """
    Groups prompt request pieces from multiple conversations into separate conversation groups.

    This function first groups pieces by conversation ID, then groups each conversation's
    pieces by sequence number. Each conversation is returned as a separate list of
    Message objects.

    Args:
        request_pieces (Sequence[MessagePiece]): A list of MessagePiece objects from
            potentially different conversations.

    Returns:
        list[list[Message]]: A list of conversations, where each conversation is a list
            of Message objects grouped by sequence.

    Example:
    >>> request_pieces = [
    >>>     MessagePiece(conversation_id="conv1", sequence=1, text="Hello"),
    >>>     MessagePiece(conversation_id="conv2", sequence=1, text="Hi there"),
    >>>     MessagePiece(conversation_id="conv1", sequence=2, text="How are you?"),
    >>>     MessagePiece(conversation_id="conv2", sequence=2, text="I'm good"),
    >>> ]
    >>> conversations = group_request_pieces_into_conversations(request_pieces)
    >>> # Returns a list of 2 conversations:
    >>> # [
    >>> #   [Message(seq=1), Message(seq=2)],  # conv1
    >>> #   [Message(seq=1), Message(seq=2)]   # conv2
    >>> # ]
    """
    if not request_pieces:
        return []

    # Group pieces by conversation ID
    conversations: dict[str, list[MessagePiece]] = {}
    for piece in request_pieces:
        conv_id = piece.conversation_id
        if conv_id not in conversations:
            conversations[conv_id] = []
        conversations[conv_id].append(piece)

    # For each conversation, group by sequence
    result: list[list[Message]] = []
    for conv_pieces in conversations.values():
        responses = group_conversation_request_pieces_by_sequence(conv_pieces)
        result.append(list(responses))

    return result


def construct_response_from_request(
    request: MessagePiece,
    response_text_pieces: list[str],
    response_type: PromptDataType = "text",
    prompt_metadata: Optional[Dict[str, Union[str, int]]] = None,
    error: PromptResponseError = "none",
) -> Message:
    """
    Constructs a response entry from a request.
    """

    if request.prompt_metadata:
        prompt_metadata = combine_dict(request.prompt_metadata, prompt_metadata or {})

    return Message(
        request_pieces=[
            MessagePiece(
                role="assistant",
                original_value=resp_text,
                conversation_id=request.conversation_id,
                labels=request.labels,
                prompt_target_identifier=request.prompt_target_identifier,
                attack_identifier=request.attack_identifier,
                original_value_data_type=response_type,
                converted_value_data_type=response_type,
                prompt_metadata=prompt_metadata,
                response_error=error,
            )
            for resp_text in response_text_pieces
        ]
    )
