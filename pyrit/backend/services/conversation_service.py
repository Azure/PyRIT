# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Conversation service for managing interactive sessions.

Handles conversation lifecycle, message sending, branching, and converter management.
"""

import importlib
import uuid
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel

from pyrit.backend.models.common import filter_sensitive_fields
from pyrit.backend.models.conversations import (
    BranchConversationRequest,
    BranchConversationResponse,
    ConverterConfig,
    CreateConversationRequest,
    CreateConversationResponse,
    MessagePieceResponse,
    MessageResponse,
    SendMessageRequest,
    SendMessageResponse,
)
from pyrit.memory import CentralMemory
from pyrit.models import Message, MessagePiece


class ConversationState(BaseModel):
    """In-memory state for an active conversation."""

    conversation_id: str
    target_class: str
    target_identifier: Dict[str, Any]
    system_prompt: Optional[str] = None
    labels: Optional[Dict[str, str]] = None
    converters: List[ConverterConfig] = []
    created_at: datetime
    message_count: int = 0


class ConversationService:
    """Service for managing conversation sessions."""

    def __init__(self) -> None:
        """Initialize the conversation service."""
        self._memory = CentralMemory.get_memory_instance()
        # In-memory conversation state (for active sessions)
        self._active_conversations: Dict[str, ConversationState] = {}
        # Instantiated converters by conversation
        self._converter_instances: Dict[str, List[Any]] = {}
        # Instantiated targets by conversation
        self._target_instances: Dict[str, Any] = {}

    def _instantiate_target_from_class(self, target_class: str, target_params: Optional[Dict[str, Any]]) -> Any:
        """
        Instantiate a target from its class name.

        Args:
            target_class: Target class name (e.g., 'TextTarget').
            target_params: Constructor parameters.

        Returns:
            Instantiated target object.
        """
        # Import the target class dynamically
        module = importlib.import_module("pyrit.prompt_target")
        cls = getattr(module, target_class, None)

        if cls is None:
            raise ValueError(f"Target class '{target_class}' not found in pyrit.prompt_target")

        params = target_params or {}
        return cls(**params)

    def _instantiate_converters(self, converter_configs: List[ConverterConfig]) -> List[Any]:
        """
        Instantiate converters from their configurations.

        Args:
            converter_configs: List of converter configurations.

        Returns:
            List of instantiated converter objects.
        """
        converters = []
        for config in converter_configs:
            module = importlib.import_module(config.module)
            converter_class = getattr(module, config.class_name)
            params = config.params or {}
            converter = converter_class(**params)
            converters.append(converter)

        return converters

    async def create_conversation(self, request: CreateConversationRequest) -> CreateConversationResponse:
        """
        Create a new conversation session.

        Args:
            request: Conversation creation request.

        Returns:
            Created conversation response with ID.
        """
        conversation_id = str(uuid.uuid4())
        now = datetime.utcnow()

        # Instantiate the target
        target = self._instantiate_target_from_class(request.target_class, request.target_params)
        self._target_instances[conversation_id] = target

        # Get the target's identifier
        target_identifier = target.get_identifier() if hasattr(target, "get_identifier") else {}

        # Store conversation state
        state = ConversationState(
            conversation_id=conversation_id,
            target_class=request.target_class,
            target_identifier=filter_sensitive_fields(target_identifier),
            labels=request.labels,
            converters=[],
            created_at=now,
        )
        self._active_conversations[conversation_id] = state

        return CreateConversationResponse(
            conversation_id=conversation_id,
            target_identifier=state.target_identifier,
            labels=state.labels,
            created_at=now,
        )

    async def get_conversation(self, conversation_id: str) -> Optional[ConversationState]:
        """
        Get conversation state by ID.

        Returns:
            Optional[ConversationState]: The conversation state or None if not found.
        """
        return self._active_conversations.get(conversation_id)

    async def get_conversation_messages(self, conversation_id: str) -> List[MessageResponse]:
        """
        Get all messages in a conversation.

        Args:
            conversation_id: The conversation ID.

        Returns:
            List of messages (grouped pieces) in order.
        """
        pieces = self._memory.get_message_pieces(conversation_id=conversation_id)

        # Sort by sequence
        pieces = sorted(pieces, key=lambda p: p.sequence)

        # Group pieces by sequence
        by_sequence: Dict[int, List[Any]] = defaultdict(list)
        for p in pieces:
            by_sequence[p.sequence].append(p)

        messages = []
        for seq in sorted(by_sequence.keys()):
            seq_pieces = by_sequence[seq]
            if not seq_pieces:
                continue

            first_piece = seq_pieces[0]
            message_pieces = [
                MessagePieceResponse(
                    id=str(p.id) if hasattr(p, "id") and p.id else str(uuid.uuid4()),
                    original_value=p.original_value or "",
                    original_value_data_type=p.original_value_data_type,
                    converted_value=p.converted_value or "",
                    converted_value_data_type=p.converted_value_data_type,
                    converter_identifiers=p.converter_identifiers or [],
                    response_error=p.response_error if hasattr(p, "response_error") else None,
                    timestamp=p.timestamp,
                )
                for p in seq_pieces
            ]

            messages.append(
                MessageResponse(
                    sequence=seq,
                    role=first_piece.role,
                    pieces=message_pieces,
                    timestamp=first_piece.timestamp,
                )
            )

        return messages

    async def send_message(
        self,
        conversation_id: str,
        request: SendMessageRequest,
    ) -> SendMessageResponse:
        """
        Send a message to the target and get a response.

        This is a simplified stub - real implementation would involve
        creating MessagePiece objects, applying converters, and calling target.

        Args:
            conversation_id: The conversation ID.
            request: Message send request.

        Returns:
            Response containing sent and received messages.
        """
        state = self._active_conversations.get(conversation_id)
        if not state:
            raise ValueError(f"Conversation {conversation_id} not found")

        target = self._target_instances.get(conversation_id)
        if not target:
            raise ValueError(f"Target for conversation {conversation_id} not found")

        now = datetime.utcnow()
        state.message_count += 1
        user_seq = state.message_count

        # Get converters if any
        converters = self._converter_instances.get(conversation_id, [])

        # Build user message pieces
        user_pieces_response = []
        user_piece_objs = []

        for piece_input in request.pieces:
            original_value = piece_input.original_value or ""
            original_type = piece_input.original_value_data_type
            converted_value = piece_input.converted_value or original_value
            converted_type = piece_input.converted_value_data_type or original_type
            converter_ids = piece_input.converter_identifiers or []

            # Apply converters if not pre-converted
            if not request.pre_converted and converters:
                for converter in converters:
                    result = await converter.convert_async(prompt=converted_value)
                    converted_value = result.output_text
                    converted_type = result.output_type
                    converter_ids.append(converter.get_identifier())

            piece_id = str(uuid.uuid4())
            user_pieces_response.append(
                MessagePieceResponse(
                    id=piece_id,
                    original_value=original_value,
                    original_value_data_type=original_type,
                    converted_value=converted_value,
                    converted_value_data_type=converted_type,
                    converter_identifiers=converter_ids,
                    response_error=None,
                    timestamp=now,
                )
            )

            # Create actual MessagePiece for target
            user_piece_objs.append(
                MessagePiece(
                    role="user",
                    original_value=original_value,
                    original_value_data_type=original_type,
                    converted_value=converted_value,
                    converted_value_data_type=converted_type,
                    converter_identifiers=converter_ids if converter_ids else None,
                    prompt_target_identifier=target.get_identifier(),
                    conversation_id=conversation_id,
                    sequence=user_seq,
                )
            )

        user_message_response = MessageResponse(
            sequence=user_seq,
            role="user",
            pieces=user_pieces_response,
            timestamp=now,
        )

        # Send to target
        user_message_obj = Message(user_piece_objs)
        response_messages = await target.send_prompt_async(message=user_message_obj)

        # Build assistant response
        assistant_message_response = None
        if response_messages:
            state.message_count += 1
            assistant_seq = state.message_count

            assistant_pieces = []
            for resp_message in response_messages:
                for resp_piece in resp_message.message_pieces:
                    assistant_pieces.append(
                        MessagePieceResponse(
                            id=str(resp_piece.id) if hasattr(resp_piece, "id") else str(uuid.uuid4()),
                            original_value=resp_piece.original_value or "",
                            original_value_data_type=resp_piece.original_value_data_type,
                            converted_value=resp_piece.converted_value or "",
                            converted_value_data_type=resp_piece.converted_value_data_type,
                            converter_identifiers=resp_piece.converter_identifiers or [],
                            response_error=getattr(resp_piece, "response_error", None),
                            timestamp=resp_piece.timestamp,
                        )
                    )

            if assistant_pieces:
                assistant_message_response = MessageResponse(
                    sequence=assistant_seq,
                    role="assistant",
                    pieces=assistant_pieces,
                    timestamp=now,
                )

        return SendMessageResponse(
            user_message=user_message_response,
            assistant_message=assistant_message_response,
        )

    async def update_system_prompt(self, conversation_id: str, system_prompt: str) -> None:
        """
        Update the system prompt for a conversation.

        Args:
            conversation_id: The conversation ID.
            system_prompt: New system prompt.
        """
        state = self._active_conversations.get(conversation_id)
        if not state:
            raise ValueError(f"Conversation {conversation_id} not found")

        target = self._target_instances.get(conversation_id)
        if not target:
            raise ValueError(f"Target for conversation {conversation_id} not found")

        # Update target system prompt
        target.set_system_prompt(
            system_prompt=system_prompt,
            conversation_id=conversation_id,
        )

        # Update state
        state.system_prompt = system_prompt

    async def update_converters(self, conversation_id: str, converters: List[ConverterConfig]) -> None:
        """
        Update the converters for a conversation.

        Args:
            conversation_id: The conversation ID.
            converters: New converter configurations.
        """
        state = self._active_conversations.get(conversation_id)
        if not state:
            raise ValueError(f"Conversation {conversation_id} not found")

        # Instantiate new converters
        converter_instances = self._instantiate_converters(converters)
        self._converter_instances[conversation_id] = converter_instances

        # Update state
        state.converters = converters

    async def branch_conversation(
        self,
        conversation_id: str,
        request: BranchConversationRequest,
    ) -> BranchConversationResponse:
        """
        Branch a conversation from a specific point.

        Args:
            conversation_id: The source conversation ID.
            request: Branch request with last_included_sequence.

        Returns:
            New conversation with copied messages.
        """
        state = self._active_conversations.get(conversation_id)
        if not state:
            raise ValueError(f"Conversation {conversation_id} not found")

        # Get messages up to branch point
        all_messages = await self.get_conversation_messages(conversation_id)
        messages_to_copy = [m for m in all_messages if m.sequence <= request.last_included_sequence]

        if not messages_to_copy:
            raise ValueError(f"No messages found at or before sequence {request.last_included_sequence}")

        # Create new conversation with same target and converters
        new_conversation_id = str(uuid.uuid4())
        now = datetime.utcnow()

        # Copy target instance
        original_target = self._target_instances.get(conversation_id)
        if original_target:
            # Create new target instance with same config
            new_target = self._instantiate_target_from_class(state.target_class, None)
            self._target_instances[new_conversation_id] = new_target

        # Copy converters
        if state.converters:
            self._converter_instances[new_conversation_id] = self._instantiate_converters(state.converters)

        # Create new state
        new_state = ConversationState(
            conversation_id=new_conversation_id,
            target_class=state.target_class,
            target_identifier=state.target_identifier,
            labels=state.labels,
            converters=state.converters,
            created_at=now,
            message_count=len(messages_to_copy),
        )
        self._active_conversations[new_conversation_id] = new_state

        # Copy messages to memory with new conversation ID
        for msg in messages_to_copy:
            for piece in msg.pieces:
                new_piece = MessagePiece(
                    role=msg.role,
                    original_value=piece.original_value,
                    original_value_data_type=piece.original_value_data_type,
                    converted_value=piece.converted_value,
                    converted_value_data_type=piece.converted_value_data_type,
                    converter_identifiers=piece.converter_identifiers if piece.converter_identifiers else None,
                    conversation_id=new_conversation_id,
                    sequence=msg.sequence,
                )
                self._memory.add_message_pieces_to_memory(message_pieces=[new_piece])

        return BranchConversationResponse(
            conversation_id=new_conversation_id,
            branched_from={
                "conversation_id": conversation_id,
                "last_included_sequence": request.last_included_sequence,
            },
            message_count=len(messages_to_copy),
            created_at=now,
        )

    async def preview_converters(
        self,
        text: str,
        converters: List[ConverterConfig],
    ) -> List[Dict[str, Any]]:
        """
        Preview text through a converter pipeline.

        Args:
            text: Input text to convert.
            converters: Converter configurations to apply.

        Returns:
            List of conversion steps showing intermediate results.
        """
        converter_instances = self._instantiate_converters(converters)

        steps = []
        current_text = text

        for i, converter in enumerate(converter_instances):
            config = converters[i]
            result = await converter.convert_async(prompt=current_text)

            steps.append(
                {
                    "step": i + 1,
                    "converter_class": config.class_name,
                    "input": current_text,
                    "output": result.output_text,
                    "output_type": result.output_type,
                }
            )

            current_text = result.output_text

        return steps

    def cleanup_conversation(self, conversation_id: str) -> None:
        """Clean up resources for a conversation."""
        self._active_conversations.pop(conversation_id, None)
        self._converter_instances.pop(conversation_id, None)
        self._target_instances.pop(conversation_id, None)


# Singleton instance
_conversation_service: Optional[ConversationService] = None


def get_conversation_service() -> ConversationService:
    """
    Get the conversation service singleton.

    Returns:
        ConversationService: The conversation service instance.
    """
    global _conversation_service
    if _conversation_service is None:
        _conversation_service = ConversationService()
    return _conversation_service
