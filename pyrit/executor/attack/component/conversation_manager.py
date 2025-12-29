# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from pyrit.memory import CentralMemory
from pyrit.models import ChatMessageRole, Message, MessagePiece, Score
from pyrit.prompt_normalizer.prompt_converter_configuration import (
    PromptConverterConfiguration,
)
from pyrit.prompt_normalizer.prompt_normalizer import PromptNormalizer
from pyrit.prompt_target import PromptTarget
from pyrit.prompt_target.common.prompt_chat_target import PromptChatTarget

logger = logging.getLogger(__name__)


def format_conversation_context(messages: List[Message]) -> str:
    """
    Format a list of messages into a context string for adversarial chat system prompts.

    This function converts conversation history into a formatted string that can be used
    by TAP and Crescendo attacks to provide context about prior conversation turns.

    For text pieces, includes both original_value and converted_value (if different).
    For non-text pieces (images, audio, etc.), uses prompt_metadata["context_description"]
    if available, otherwise uses a placeholder like [Image] or [Audio].

    Args:
        messages (List[Message]): The conversation messages to format.

    Returns:
        str: A formatted string representing the conversation context.
            Returns empty string if no messages provided.

    Example output:
        Turn 1:
        User: How do I make a cake?
        Assistant: Here's a simple recipe for making a cake...

        Turn 2:
        User: [Image - A photo of baking ingredients]
        Assistant: I can see flour, eggs, and sugar in your image...
    """
    if not messages:
        return ""

    context_parts: List[str] = []
    turn_number = 0

    for message in messages:
        piece = message.get_piece()

        # Skip system messages - they're handled separately
        if piece.role == "system":
            continue

        # Start a new turn when we see a user message
        if piece.role == "user":
            turn_number += 1
            context_parts.append(f"Turn {turn_number}:")

        # Format the piece content
        content = _format_piece_content(piece)
        role_label = "User" if piece.role == "user" else "Assistant"
        context_parts.append(f"{role_label}: {content}")

    return "\n".join(context_parts)


def _format_piece_content(piece: MessagePiece) -> str:
    """
    Format a single message piece into a content string.

    For text pieces, shows original and converted values (if different).
    For non-text pieces, uses context_description metadata or a placeholder.

    Args:
        piece (MessagePiece): The message piece to format.

    Returns:
        str: The formatted content string.
    """
    data_type = piece.converted_value_data_type or piece.original_value_data_type

    # For non-text pieces, use metadata description or placeholder
    if data_type != "text":
        # Check for context_description in metadata
        if piece.prompt_metadata and "context_description" in piece.prompt_metadata:
            description = piece.prompt_metadata["context_description"]
            return f"[{data_type.capitalize()} - {description}]"
        else:
            return f"[{data_type.capitalize()}]"

    # For text pieces, include both original and converted if different
    original = piece.original_value
    converted = piece.converted_value

    if original != converted:
        return f"{converted} (original: {original})"
    else:
        return converted


@dataclass
class ConversationState:
    """Container for conversation state data shared between attack components."""

    turn_count: int = 0
    last_user_message: str = ""
    last_assistant_message_scores: List[Score] = field(default_factory=list)


class ConversationManager:
    """
    Manages conversations for attacks, handling message history,
    system prompts, and conversation state.
    This class provides methods to retrieve conversations, add system prompts,
    and update conversation state with prepended messages.
    """

    def __init__(
        self,
        *,
        attack_identifier: dict[str, str],
        prompt_normalizer: Optional[PromptNormalizer] = None,
    ):
        """
        Initialize the conversation manager.

        Args:
            attack_identifier (dict[str, str]): The identifier of the attack this manager belongs to.
            prompt_normalizer (Optional[PromptNormalizer]): Optional prompt normalizer to use for converting prompts.
                If not provided, a default PromptNormalizer instance will be created.
        """
        self._prompt_normalizer = prompt_normalizer or PromptNormalizer()
        self._memory = CentralMemory.get_memory_instance()
        self._attack_identifier = attack_identifier

    def get_conversation(self, conversation_id: str) -> List[Message]:
        """
        Retrieve a conversation by its ID.

        Args:
            conversation_id (str): The ID of the conversation to retrieve.

        Returns:
            List[Message]: A list of messages in the conversation, ordered by their creation time.
                If no messages exist, an empty list is returned.
        """
        conversation = self._memory.get_conversation(conversation_id=conversation_id)
        return list(conversation)

    def get_last_message(
        self, *, conversation_id: str, role: Optional[ChatMessageRole] = None
    ) -> Optional[MessagePiece]:
        """
        Retrieve the most recent message from a conversation.

        Args:
            conversation_id (str): The ID of the conversation to retrieve the last message from.
            role (Optional[ChatMessageRole]): If provided, only return the last message that matches this role.

        Returns:
            Optional[MessagePiece]: The last message piece from the conversation,
                or `None` if no messages exist.
        """
        conversation = self.get_conversation(conversation_id)
        if not conversation:
            return None

        if role:
            for m in reversed(conversation):
                piece = m.get_piece()
                if piece.role == role:
                    return piece
            return None

        return conversation[-1].get_piece()

    def set_system_prompt(
        self,
        *,
        target: PromptChatTarget,
        conversation_id: str,
        system_prompt: str,
        labels: Optional[Dict[str, str]] = None,
    ) -> None:
        """
        Set or update the system-level prompt associated with a conversation.

        This helper is intended for conversational (`PromptChatTarget`) goals,
        where a dedicated system prompt influences the behavior of the LLM for
        all subsequent user / assistant messages in the same `conversation_id`.

        Args:
            target (PromptChatTarget): The target to set the system prompt on.
            conversation_id (str): Unique identifier for the conversation to set the system prompt on.
            system_prompt (str): The system prompt to set for the conversation.
            labels (Optional[Dict[str, str]]): Optional labels to associate with the system prompt.
                These can be used for categorization or filtering purposes.
        """
        target.set_system_prompt(
            system_prompt=system_prompt,
            conversation_id=conversation_id,
            attack_identifier=self._attack_identifier,
            labels=labels,
        )

    async def update_conversation_state_async(
        self,
        *,
        target: PromptTarget,
        conversation_id: str,
        prepended_conversation: List[Message],
        request_converters: Optional[List[PromptConverterConfiguration]] = None,
        response_converters: Optional[List[PromptConverterConfiguration]] = None,
        max_turns: Optional[int] = None,
    ) -> ConversationState:
        """
        Prepare a chat conversation by attaching history, enforcing
        target-specific rules, optionally normalizing prompts, and returning a
        serializable `ConversationState`.

        This helper is designed to support two distinct usage patterns:

        Single-turn bootstrap - When `max_turns` is **not** supplied the function simply injects the
        provided `prepended_conversation` into memory, performs any requested
        prompt conversions, and exits.

        Multi-turn continuation - When `max_turns` **is** supplied the function acts as a state machine:
        it verifies that the running history does not exceed the allowed turn budget, excludes
        the most recent user-utterance (so that an attack can re-inject it as the "live" request),
        and extracts per-session counters such as the current turn index.

        Args:
            target (PromptTarget): The target for which the conversation is being prepared.
                Used to validate that prepended_conversation is compatible with the target type.
            conversation_id (str): Unique identifier for the conversation to update or create.
            prepended_conversation (List[Message]):
                List of messages to prepend to the conversation history.
            request_converters (Optional[List[PromptConverterConfiguration]]):
                List of configurations for converting user (request) messages.
            response_converters (Optional[List[PromptConverterConfiguration]]):
                List of configurations for converting assistant (response) messages.
            max_turns (Optional[int]): Maximum number of turns allowed in the conversation. If not provided,
                the function assumes a single-turn context.

        Returns:
            ConversationState: A snapshot of the conversation state after processing the prepended
                messages, including turn count and last user message.

        Raises:
            ValueError: If `conversation_id` is empty, if the last message in a multi-turn
                context is a user message (which should not be prepended), or if
                prepended_conversation is provided with a non-PromptChatTarget target.
        """
        if not conversation_id:
            raise ValueError("conversation_id cannot be empty")

        # Validate prepended_conversation compatibility with target type
        # Non-chat targets do not read conversation history from memory
        if prepended_conversation and not isinstance(target, PromptChatTarget):
            raise ValueError(
                "prepended_conversation requires target to be a PromptChatTarget. "
                "Non-chat targets do not support explicit conversation history management."
            )

        # Initialize conversation state
        state = ConversationState()
        logger.debug(f"Preparing conversation with ID: {conversation_id}")

        # Do not proceed if no history is provided
        if not prepended_conversation:
            logger.debug(f"No history provided for conversation initialization: {conversation_id}")
            return state

        # Filter out None values and empty requests
        valid_requests = [req for req in prepended_conversation if req is not None and req.message_pieces]

        if not valid_requests:
            logger.debug(f"No valid requests in prepended conversation for: {conversation_id}")
            return state

        # Determine if we should exclude the last message (if it's a user message in multi-turn context)
        last_message = valid_requests[-1].message_pieces[0]
        is_multi_turn = max_turns is not None
        should_exclude_last = is_multi_turn and last_message.role == "user"

        # Process all messages except potentially the last one
        for i, request in enumerate(valid_requests):
            # Skip the last message if it's a user message in multi-turn context
            if should_exclude_last and i == len(valid_requests) - 1:
                logger.debug("Skipping last user message (will be added by attack)")
                continue

            # Apply converters if needed
            if request_converters or response_converters:
                logger.debug(f"Converting request {i + 1}/{len(valid_requests)} in conversation {conversation_id}")
                # Apply role-specific converters
                await self._apply_role_specific_converters_async(
                    request=request,
                    request_converters=request_converters,
                    response_converters=response_converters,
                )

            # Process the message piece
            logger.debug(f"Processing message {i + 1}/{len(valid_requests)} in conversation {conversation_id}")
            await self._process_prepended_message_async(
                request=request,
                conversation_id=conversation_id,
                conversation_state=state,
                max_turns=max_turns,
            )

        # Extract state from the conversation history (only for multi-turn conversations)
        if is_multi_turn:
            await self._populate_conversation_state_async(
                last_message=last_message,
                prepended_conversation=valid_requests,
                conversation_state=state,
            )

        return state

    async def _apply_role_specific_converters_async(
        self,
        *,
        request: Message,
        request_converters: Optional[List[PromptConverterConfiguration]] = None,
        response_converters: Optional[List[PromptConverterConfiguration]] = None,
    ) -> None:
        """
        Apply role-specific converters to messages.

        - Request converters are applied to 'user' role messages
        - Response converters are applied to 'assistant' role messages
        - No converters are applied to 'system' role messages

        Args:
            request (Message): The request containing pieces to convert.
            request_converters (Optional[List[PromptConverterConfiguration]]):
                Converter configurations to apply to 'user' role messages.
            response_converters (Optional[List[PromptConverterConfiguration]]):
                Converter configurations to apply to 'assistant' role messages.
        """
        # Determine which converters to apply based on message roles
        for piece in request.message_pieces:
            applicable_converters: Optional[List[PromptConverterConfiguration]] = None

            if piece.role == "user" and request_converters:
                applicable_converters = request_converters
            elif piece.role == "assistant" and response_converters:
                applicable_converters = response_converters
            # System messages get no converters (applicable_converters remains None)

            # Apply the determined converters
            if applicable_converters:
                # Create a temporary request with just this piece for conversion
                temp_request = Message(message_pieces=[piece])
                await self._prompt_normalizer.convert_values(
                    message=temp_request,
                    converter_configurations=applicable_converters,
                )

    async def _process_prepended_message_async(
        self,
        *,
        request: Message,
        conversation_id: str,
        conversation_state: ConversationState,
        max_turns: Optional[int] = None,
    ) -> None:
        """
        Process a prepended message and update the conversation state.
        This method handles the conversion of message pieces, sets conversation IDs,
        and attack identifiers, and processes each piece based on its role.

        Args:
            request (Message): The request containing pieces to process.
            conversation_id (str): The ID of the conversation to update.
            conversation_state (ConversationState): The current state of the conversation.
            max_turns (Optional[int]): Maximum allowed turns for the conversation.
        """
        # Validate the request before processing
        if not request or not request.message_pieces:
            return

        # Set the conversation ID and attack ID for each piece in the request
        for piece in request.message_pieces:
            piece.conversation_id = conversation_id
            piece.attack_identifier = self._attack_identifier
            piece.id = uuid.uuid4()

            # Process the piece based on its role (validates turn count for multi-turn)
            self._process_piece(
                piece=piece,
                conversation_state=conversation_state,
                max_turns=max_turns,
            )

        # Add the request to memory
        self._memory.add_message_to_memory(request=request)

    def _process_piece(
        self,
        *,
        piece: MessagePiece,
        conversation_state: ConversationState,
        max_turns: Optional[int] = None,
    ) -> None:
        """
        Process a message piece based on its role and update conversation state.

        For multi-turn conversations, this validates that the turn count doesn't exceed
        max_turns. Only assistant messages count as turns.

        Args:
            piece (MessagePiece): The piece to process.
            conversation_state (ConversationState): The current state of the conversation.
            max_turns (Optional[int]): Maximum allowed turns (for validation).

        Raises:
            ValueError: If max_turns would be exceeded by this piece.
        """
        is_multi_turn = max_turns is not None

        # Only assistant messages count as turns
        if piece.role == "assistant" and is_multi_turn:
            conversation_state.turn_count += 1

            if conversation_state.turn_count > max_turns:
                raise ValueError(
                    f"The number of turns in the prepended conversation ({conversation_state.turn_count-1}) is equal to"
                    + f" or exceeds the maximum number of turns ({max_turns}), which means the"
                    + " conversation will not be able to continue. Please reduce the number of turns in"
                    + " the prepended conversation or increase the maximum number of turns and try again."
                )

    async def _populate_conversation_state_async(
        self,
        *,
        prepended_conversation: List[Message],
        last_message: MessagePiece,
        conversation_state: ConversationState,
    ) -> None:
        """
        Extract conversation context from the last messages in prepended_conversation.

        This extracts:
        - Last user message for continuing conversations.
        - Scores for the last assistant message for evaluation.

        Args:
            prepended_conversation (List[Message]): Complete conversation history.
            last_message (MessagePiece): The last message in the history.
            conversation_state (ConversationState): State object to populate.

        Raises:
            ValueError: If an assistant message doesn't have a preceding user message.
        """
        if not prepended_conversation:
            return  # Nothing to extract from empty history

        # Extract the last user message and assistant message scores from the last message
        if last_message.role == "user":
            conversation_state.last_user_message = last_message.converted_value
            logger.debug(f"Extracted last user message: {conversation_state.last_user_message[:50]}...")

        elif last_message.role == "assistant":
            # Get scores for the last assistant message based off of the original id
            conversation_state.last_assistant_message_scores = list(
                self._memory.get_prompt_scores(prompt_ids=[str(last_message.original_prompt_id)])
            )

            # Do not set last user message if there are no scores for the last assistant message
            if not conversation_state.last_assistant_message_scores:
                logger.debug("No scores found for last assistant message")
                return

            # Check assumption that there will be a user message preceding the assistant message
            if len(prepended_conversation) > 1 and prepended_conversation[-2].get_piece().role == "user":
                conversation_state.last_user_message = prepended_conversation[-2].get_value()
                logger.debug(f"Extracted preceding user message: {conversation_state.last_user_message[:50]}...")
            else:
                raise ValueError(
                    "There must be a user message preceding the assistant message in prepended conversations."
                )

    async def prepend_to_adversarial_chat_async(
        self,
        *,
        adversarial_chat: PromptChatTarget,
        adversarial_chat_conversation_id: str,
        prepended_conversation: List[Message],
        labels: Optional[Dict[str, str]] = None,
    ) -> None:
        """
        Replay prepended conversation to the adversarial chat's memory with swapped roles.

        This method takes a conversation history (typically between user and objective target)
        and replays it to the adversarial chat so it has context of the established conversation.
        Roles are swapped because from the adversarial chat's perspective:
        - "user" messages in the original become "assistant" (what the adversarial chat said)
        - "assistant" messages become "user" (responses it received)

        This is useful when using prepended_conversation to establish context (e.g., role-play
        scenarios) and wanting the adversarial chat to continue naturally from that context.

        Args:
            adversarial_chat (PromptChatTarget): The adversarial chat target to prepend to.
            adversarial_chat_conversation_id (str): The conversation ID for the adversarial chat.
            prepended_conversation (List[Message]): The conversation history to replay.
            labels (Optional[Dict[str, str]]): Optional labels to associate with the messages.

        Note:
            - System messages are skipped (adversarial chat has its own system prompt)
            - Messages are added to memory directly without LLM calls
        """
        if not prepended_conversation:
            logger.debug("No prepended conversation to replay to adversarial chat")
            return

        # Role mapping: swap user <-> assistant for adversarial chat's perspective
        role_swap: Dict[ChatMessageRole, ChatMessageRole] = {
            "user": "assistant",
            "assistant": "user",
        }

        for message in prepended_conversation:
            for piece in message.message_pieces:
                # Skip system messages - adversarial chat has its own system prompt
                if piece.role == "system":
                    continue

                # Create a new piece with swapped role for adversarial chat
                swapped_role = role_swap.get(piece.role, piece.role)

                adversarial_piece = MessagePiece(
                    id=uuid.uuid4(),
                    role=swapped_role,
                    original_value=piece.original_value,
                    converted_value=piece.converted_value,
                    original_value_data_type=piece.original_value_data_type,
                    converted_value_data_type=piece.converted_value_data_type,
                    conversation_id=adversarial_chat_conversation_id,
                    attack_identifier=self._attack_identifier,
                    prompt_target_identifier=adversarial_chat.get_identifier(),
                    labels=labels,
                )

                # Add to memory
                self._memory.add_message_to_memory(request=adversarial_piece.to_message())

        logger.debug(
            f"Replayed {len(prepended_conversation)} messages to adversarial chat "
            f"conversation {adversarial_chat_conversation_id}"
        )
