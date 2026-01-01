# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from pyrit.memory import CentralMemory
from pyrit.models import ChatMessageRole, Message, MessagePiece, Score
from pyrit.prompt_normalizer.prompt_converter_configuration import (
    PromptConverterConfiguration,
)
from pyrit.prompt_normalizer.prompt_normalizer import PromptNormalizer
from pyrit.prompt_target import PromptTarget
from pyrit.prompt_target.common.prompt_chat_target import PromptChatTarget

logger = logging.getLogger(__name__)


def mark_messages_as_simulated(messages: Sequence[Message]) -> List[Message]:
    """
    Mark assistant messages as simulated_assistant for traceability.

    This function converts all assistant roles to simulated_assistant in the
    provided messages. This is useful when loading conversations from YAML files
    or other sources where the responses are not from actual targets.

    Args:
        messages (Sequence[Message]): The messages to mark as simulated.

    Returns:
        List[Message]: The same messages with assistant roles converted to simulated_assistant.
            Modifies the messages in place and also returns them for convenience.
    """
    result = list(messages)
    for message in result:
        for piece in message.message_pieces:
            if piece._role == "assistant":
                piece._role = "simulated_assistant"
    return result


def format_conversation_context(messages: List[Message]) -> str:
    """
    Format a list of messages into a context string for adversarial chat system prompts.

    This function converts conversation history into a formatted string that can be used
    by TAP and Crescendo attacks to provide context about prior conversation turns.

    For text pieces, includes both original_value and converted_value (if different).
    For non-text pieces (images, audio, etc.), uses prompt_metadata["context_description"]
    if available, otherwise uses a placeholder like [Image] or [Audio].

    Args:
        messages: The conversation messages to format.

    Returns:
        A formatted string representing the conversation context.
        Returns empty string if no messages provided.
    """
    if not messages:
        return ""

    context_parts: List[str] = []
    turn_number = 0

    for message in messages:
        piece = message.get_piece()

        # Skip system messages - they're handled separately
        if piece.api_role == "system":
            continue

        # Start a new turn when we see a user message
        if piece.api_role == "user":
            turn_number += 1
            context_parts.append(f"Turn {turn_number}:")

        # Format the piece content
        content = _format_piece_content(piece)
        if piece.api_role == "user":
            role_label = "User"
        elif piece.is_simulated:
            role_label = "Assistant (simulated)"
        else:
            role_label = "Assistant"
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

    # Scores from the last assistant message (for attack-specific interpretation)
    last_assistant_message_scores: List[Score] = field(default_factory=list)

    # The last unanswered user message (preserved as original Message for multimodal support)
    last_unanswered_user_message: Optional[Message] = None

    # Normalized prepended conversation context for the objective target.
    # Set when non_chat_target_behavior="normalize_first_turn" is configured.
    # This context should be prepended to the first message sent to the target.
    normalized_prepended_context: Optional[str] = None

    # Normalized context string for adversarial chat system prompts
    adversarial_chat_context: Optional[str] = None


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
                if piece.api_role == role:
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

    async def apply_prepended_conversation_to_objective_async(
        self,
        *,
        target: PromptTarget,
        conversation_id: str,
        prepended_conversation: List[Message],
        request_converters: Optional[List[PromptConverterConfiguration]] = None,
        max_turns: Optional[int] = None,
        prepended_conversation_config: Optional["PrependedConversationConfiguration"] = None,
    ) -> ConversationState:
        """
        Apply prepended conversation to the objective target's conversation history.

        For PromptChatTarget: Adds messages directly to the conversation memory.
        For non-chat targets: Normalizes conversation to a string stored in
        ConversationState.normalized_prepended_context for inclusion in the first message.

        Args:
            target (PromptTarget): The objective target for the conversation.
            conversation_id (str): Unique identifier for the conversation.
            prepended_conversation (List[Message]): Messages to prepend to the conversation.
            request_converters (Optional[List[PromptConverterConfiguration]]):
                Converters to apply to user messages before adding to memory.
            max_turns (Optional[int]): Maximum turns allowed. When provided, validates
                turn count and extracts state for multi-turn attacks.
            prepended_conversation_config (Optional[PrependedConversationConfiguration]):
                Configuration for converter application and non-chat target behavior.

        Returns:
            ConversationState: State containing turn_count, last_user_message,
                and normalized_prepended_context (for non-chat targets).

        Raises:
            ValueError: If conversation_id is empty, or if prepended_conversation is
                provided with a non-chat target and behavior is "raise".
        """
        if not conversation_id:
            raise ValueError("conversation_id cannot be empty")

        # Initialize conversation state
        state = ConversationState()
        logger.debug(f"Preparing conversation with ID: {conversation_id}")

        # Do not proceed if no history is provided
        if not prepended_conversation:
            logger.debug(f"No history provided for conversation initialization: {conversation_id}")
            return state

        # Validate prepended_conversation compatibility with target type
        # Non-chat targets do not read conversation history from memory
        is_chat_target = isinstance(target, PromptChatTarget)
        if prepended_conversation and not is_chat_target:
            # Check configuration for how to handle non-chat targets
            behavior = (
                prepended_conversation_config.non_chat_target_behavior
                if prepended_conversation_config
                else "raise"
            )
            if behavior == "raise":
                raise ValueError(
                    "prepended_conversation requires target to be a PromptChatTarget. "
                    "Non-chat targets do not support explicit conversation history management. "
                    "Use PrependedConversationConfiguration with non_chat_target_behavior='normalize_first_turn' "
                    "to normalize the conversation into the first message instead."
                )
            elif behavior == "normalize_first_turn":
                # Normalize the prepended conversation into a string for inclusion in first message
                normalized_context = await self._normalize_prepended_conversation_async(
                    prepended_conversation=prepended_conversation,
                    config=prepended_conversation_config,
                )
                state.normalized_prepended_context = normalized_context
                logger.debug(
                    f"Normalized prepended conversation for non-chat target: "
                    f"{len(normalized_context)} characters"
                )
                return state

        # Filter out None values and empty requests
        valid_requests = [req for req in prepended_conversation if req is not None and req.message_pieces]

        if not valid_requests:
            logger.debug(f"No valid requests in prepended conversation for: {conversation_id}")
            return state

        # Determine if we should exclude the last message (if it's a user message in multi-turn context)
        last_message = valid_requests[-1].message_pieces[0]
        is_multi_turn = max_turns is not None
        should_exclude_last = is_multi_turn and last_message.api_role == "user"

        # Process all messages except potentially the last one
        for i, request in enumerate(valid_requests):
            # Skip the last message if it's a user message in multi-turn context
            if should_exclude_last and i == len(valid_requests) - 1:
                logger.debug("Skipping last user message (will be added by attack)")
                continue

            # Apply converters if needed, respecting config's apply_converters_to_roles
            if request_converters:
                logger.debug(f"Converting request {i + 1}/{len(valid_requests)} in conversation {conversation_id}")
                # Apply converters with optional role filtering from config
                await self._apply_converters_async(
                    request=request,
                    request_converters=request_converters,
                    apply_to_roles=(
                        prepended_conversation_config.apply_converters_to_roles
                        if prepended_conversation_config
                        else None
                    ),
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
                last_user_message=valid_requests[-1] if should_exclude_last else None,
                prepended_conversation=valid_requests,
                conversation_state=state,
            )

        return state

    async def initialize_prepended_conversation_async(
        self,
        *,
        context: "AttackContext",
        target: PromptTarget,
        conversation_id: str,
        max_turns: Optional[int] = None,
        request_converters: Optional[List[PromptConverterConfiguration]] = None,
        prepended_conversation_config: Optional["PrependedConversationConfiguration"] = None,
    ) -> ConversationState:
        """
        Initialize prepended conversation and update the attack context directly.

        This method processes the prepended conversation from context, adds messages
        to memory, and updates context fields:
        - context.executed_turns: Updated with the turn count from prepended conversation
        - context.next_message: Set to the last unanswered user message (if any) when
          context.next_message is None, preserving the original Message for multimodal support

        Args:
            context (AttackContext): The attack context to update.
            target (PromptTarget): The objective target for the conversation.
            conversation_id (str): Unique identifier for the conversation.
            max_turns (Optional[int]): Maximum turns allowed. When provided, validates
                turn count and extracts state for multi-turn attacks.
            request_converters (Optional[List[PromptConverterConfiguration]]):
                Converters to apply to messages before adding to memory.
            prepended_conversation_config (Optional[PrependedConversationConfiguration]):
                Configuration for converter application and non-chat target behavior.

        Returns:
            ConversationState: State containing last_assistant_message_scores and
                normalized_prepended_context (for non-chat targets). Attacks can use
                the scores for attack-specific logic (e.g., refusal detection).
        """
        # Call the existing method to do the work
        state = await self.apply_prepended_conversation_to_objective_async(
            target=target,
            conversation_id=conversation_id,
            prepended_conversation=context.prepended_conversation,
            request_converters=request_converters,
            max_turns=max_turns,
            prepended_conversation_config=prepended_conversation_config,
        )

        # Update context.executed_turns for multi-turn attacks
        if hasattr(context, "executed_turns"):
            context.executed_turns = state.turn_count

        # If there's an unanswered user message and context.next_message is not set,
        # preserve the original Message (not just text) for multimodal support
        if state.last_unanswered_user_message is not None and context.next_message is None:
            context.next_message = state.last_unanswered_user_message
            logger.debug("Set context.next_message to last unanswered user message from prepended conversation")

        return state

    async def _apply_converters_async(
        self,
        *,
        request: Message,
        request_converters: List[PromptConverterConfiguration],
        apply_to_roles: Optional[List[ChatMessageRole]] = None,
    ) -> None:
        """
        Apply converters to messages in the request.

        Args:
            request (Message): The request containing pieces to convert.
            request_converters (List[PromptConverterConfiguration]):
                Converter configurations to apply to messages.
            apply_to_roles (Optional[List[ChatMessageRole]]):
                If provided, only apply converters to messages with roles in this list.
                If None, applies to all roles.
        """
        for piece in request.message_pieces:
            # If apply_to_roles is specified, only apply to those roles
            # If None, apply to all roles (no filtering)
            if apply_to_roles is not None and piece.role not in apply_to_roles:
                continue

            temp_request = Message(message_pieces=[piece])
            await self._prompt_normalizer.convert_values(
                message=temp_request,
                converter_configurations=request_converters,
            )

    async def _normalize_prepended_conversation_async(
        self,
        *,
        prepended_conversation: List[Message],
        config: Optional["PrependedConversationConfiguration"],
    ) -> str:
        """
        Normalize a prepended conversation into a single text string for the objective target.

        This method uses the configured normalizer to convert the messages into a string.
        If no normalizer is configured, it uses ConversationContextNormalizer as the
        default for basic "Turn N: User/Assistant" formatting.

        Args:
            prepended_conversation: The list of messages to normalize.
            config: The prepended conversation configuration with the normalizer.

        Returns:
            A string representation of the normalized conversation.
        """
        if not prepended_conversation:
            return ""

        # Use the configured normalizer via helper method (handles default)
        if config:
            normalizer = config.get_objective_target_normalizer()
        else:
            normalizer = ConversationContextNormalizer()

        return await normalizer.normalize_string_async(prepended_conversation)

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
        if piece.api_role == "assistant" and is_multi_turn:
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
        last_user_message: Optional[Message],
        conversation_state: ConversationState,
    ) -> None:
        """
        Extract conversation context from the last messages in prepended_conversation.

        This extracts:
        - Last unanswered user message (as original Message for multimodal support).
        - Scores for the last assistant message for evaluation.

        Args:
            prepended_conversation (List[Message]): Complete conversation history.
            last_message (MessagePiece): The last message piece in the history.
            last_user_message (Optional[Message]): The last user message if it was excluded
                from memory (unanswered). Preserved as-is for multimodal support.
            conversation_state (ConversationState): State object to populate.

        Raises:
            ValueError: If an assistant message doesn't have a preceding user message.
        """
        if not prepended_conversation:
            return  # Nothing to extract from empty history

        # If last message is a user message that was excluded, preserve the original Message
        if last_message.api_role == "user" and last_user_message is not None:
            conversation_state.last_unanswered_user_message = last_user_message
            logger.debug(f"Preserved last unanswered user message: {last_message.converted_value[:50]}...")

        elif last_message.api_role == "assistant":
            # Get scores for the last assistant message based off of the original id
            conversation_state.last_assistant_message_scores = list(
                self._memory.get_prompt_scores(prompt_ids=[str(last_message.original_prompt_id)])
            )

            if not conversation_state.last_assistant_message_scores:
                logger.debug("No scores found for last assistant message")
                return

            # Check assumption that there will be a user message preceding the assistant message
            if len(prepended_conversation) > 1 and prepended_conversation[-2].get_piece().api_role == "user":
                conversation_state.last_user_message = prepended_conversation[-2].get_value()
                logger.debug(f"Extracted preceding user message: {conversation_state.last_user_message[:50]}...")
            else:
                raise ValueError(
                    "There must be a user message preceding the assistant message in prepended conversations."
                )

    async def apply_prepended_conversation_to_adversarial_async(
        self,
        *,
        adversarial_chat: PromptChatTarget,
        adversarial_chat_conversation_id: str,
        prepended_conversation: List[Message],
        state: Optional[ConversationState] = None,
        prepended_conversation_config: Optional["PrependedConversationConfiguration"] = None,
        labels: Optional[Dict[str, str]] = None,
    ) -> ConversationState:
        """
        Apply prepended conversation to the adversarial chat for multi-turn attacks.

        This method performs two operations:
        1. Builds a context string using the configured normalizer and stores it in
           state.adversarial_chat_context for use in system prompts.
        2. Replays the conversation to adversarial chat memory with swapped roles
           (user↔assistant) so the adversarial chat has conversation history context.

        Args:
            adversarial_chat (PromptChatTarget): The adversarial chat target.
            adversarial_chat_conversation_id (str): Conversation ID for the adversarial chat.
            prepended_conversation (List[Message]): The conversation history to apply.
            state (Optional[ConversationState]): Existing state to update. If None, creates new state.
            prepended_conversation_config (Optional[PrependedConversationConfiguration]):
                Configuration with adversarial_chat_context_normalizer for formatting context.
            labels (Optional[Dict[str, str]]): Labels to associate with the messages.

        Returns:
            ConversationState: Updated state with adversarial_chat_context populated.

        Note:
            - System messages are skipped (adversarial chat has its own system prompt)
            - Messages are added to memory directly without LLM calls
            - Roles are swapped: user→assistant, assistant→user
        """
        if state is None:
            state = ConversationState()

        if not prepended_conversation:
            logger.debug("No prepended conversation to apply to adversarial chat")
            return state

        # Build context string for system prompt using configured normalizer
        if prepended_conversation_config:
            normalizer = prepended_conversation_config.get_adversarial_chat_normalizer()
        else:
            normalizer = ConversationContextNormalizer()

        state.adversarial_chat_context = await normalizer.normalize_string_async(prepended_conversation)
        logger.debug(f"Built adversarial chat context: {len(state.adversarial_chat_context)} characters")

        # Replay messages to adversarial chat memory with swapped roles
        await self._replay_to_adversarial_chat_async(
            adversarial_chat=adversarial_chat,
            adversarial_chat_conversation_id=adversarial_chat_conversation_id,
            prepended_conversation=prepended_conversation,
            labels=labels,
        )

        return state

    async def _replay_to_adversarial_chat_async(
        self,
        *,
        adversarial_chat: PromptChatTarget,
        adversarial_chat_conversation_id: str,
        prepended_conversation: List[Message],
        labels: Optional[Dict[str, str]] = None,
    ) -> None:
        """
        Replay prepended conversation to adversarial chat memory with swapped roles.

        Roles are swapped because from the adversarial chat's perspective:
        - "user" messages become "assistant" (prompts it generated)
        - "assistant" messages become "user" (responses it received)

        Args:
            adversarial_chat (PromptChatTarget): The adversarial chat target.
            adversarial_chat_conversation_id (str): Conversation ID for the adversarial chat.
            prepended_conversation (List[Message]): The conversation history to replay.
            labels (Optional[Dict[str, str]]): Labels to associate with the messages.
        """
        role_swap: Dict[ChatMessageRole, ChatMessageRole] = {
            "user": "assistant",
            "assistant": "user",
        }

        for message in prepended_conversation:
            for piece in message.message_pieces:
                # Skip system messages - adversarial chat has its own system prompt
                if piece.api_role == "system":
                    continue

                # Create a new piece with swapped role for adversarial chat
                swapped_role = role_swap.get(piece.api_role, piece.api_role)

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

                self._memory.add_message_to_memory(request=adversarial_piece.to_message())

        logger.debug(
            f"Replayed {len(prepended_conversation)} messages to adversarial chat "
            f"conversation {adversarial_chat_conversation_id}"
        )
