# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union

from pyrit.memory import CentralMemory
from pyrit.models import PromptRequestPiece, PromptRequestResponse
from pyrit.models.literals import ChatMessageRole
from pyrit.models.score import Score
from pyrit.prompt_normalizer.prompt_converter_configuration import (
    PromptConverterConfiguration,
)
from pyrit.prompt_normalizer.prompt_normalizer import PromptNormalizer
from pyrit.prompt_target.common.prompt_chat_target import PromptChatTarget
from pyrit.prompt_target.common.prompt_target import PromptTarget

logger = logging.getLogger(__name__)


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

    def get_conversation(self, conversation_id: str) -> List[PromptRequestResponse]:
        """
        Retrieve a conversation by its ID.

        Args:
            conversation_id (str): The ID of the conversation to retrieve.

        Returns:
            List[PromptRequestResponse]: A list of messages in the conversation, ordered by their creation time.
                If no messages exist, an empty list is returned.
        """
        conversation = self._memory.get_conversation(conversation_id=conversation_id)
        return list(conversation)

    def get_last_message(
        self, *, conversation_id: str, role: Optional[ChatMessageRole] = None
    ) -> Optional[PromptRequestPiece]:
        """
        Retrieve the most recent message from a conversation.

        Args:
            conversation_id (str): The ID of the conversation to retrieve the last message from.
            role (Optional[ChatMessageRole]): If provided, only return the last message that matches this role.

        Returns:
            Optional[PromptRequestPiece]: The last message piece from the conversation,
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
        set or update the system-level prompt associated with a conversation.

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
        conversation_id: str,
        target: Optional[Union[PromptTarget, PromptChatTarget]] = None,
        prepended_conversation: List[PromptRequestResponse],
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
            conversation_id (str): Unique identifier for the conversation to update or create.
            target (Optional[Union[PromptTarget, PromptChatTarget]]): The target to set system prompts on (if
                applicable).
            prepended_conversation (List[PromptRequestResponse]):
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
            ValueError: If `conversation_id` is empty or if the last message in a multi-turn
                context is a user message (which should not be prepended).
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

        # Filter out None values and empty requests
        valid_requests = [req for req in prepended_conversation if req is not None and req.request_pieces]

        if not valid_requests:
            logger.debug(f"No valid requests in prepended conversation for: {conversation_id}")
            return state

        # Determine if we should exclude the last message (if it's a user message in multi-turn context)
        last_message = valid_requests[-1].request_pieces[0]
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

            # Process the request piece
            logger.debug(f"Processing message {i + 1}/{len(valid_requests)} in conversation {conversation_id}")
            await self._process_prepended_message_async(
                request=request,
                conversation_id=conversation_id,
                conversation_state=state,
                target=target,
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
        request: PromptRequestResponse,
        request_converters: Optional[List[PromptConverterConfiguration]] = None,
        response_converters: Optional[List[PromptConverterConfiguration]] = None,
    ) -> None:
        """
        Apply role-specific converters to messages.

        - Request converters are applied to 'user' role messages
        - Response converters are applied to 'assistant' role messages
        - No converters are applied to 'system' role messages

        Args:
            request (PromptRequestResponse): The request containing pieces to convert.
            request_converters (Optional[List[PromptConverterConfiguration]]):
                Converter configurations to apply to 'user' role messages.
            response_converters (Optional[List[PromptConverterConfiguration]]):
                Converter configurations to apply to 'assistant' role messages.
        """
        # Determine which converters to apply based on message roles
        for piece in request.request_pieces:
            applicable_converters: Optional[List[PromptConverterConfiguration]] = None

            if piece.role == "user" and request_converters:
                applicable_converters = request_converters
            elif piece.role == "assistant" and response_converters:
                applicable_converters = response_converters
            # System messages get no converters (applicable_converters remains None)

            # Apply the determined converters
            if applicable_converters:
                # Create a temporary request with just this piece for conversion
                temp_request = PromptRequestResponse(request_pieces=[piece])
                await self._prompt_normalizer.convert_values(
                    request_response=temp_request,
                    converter_configurations=applicable_converters,
                )

    async def _process_prepended_message_async(
        self,
        *,
        request: PromptRequestResponse,
        conversation_id: str,
        conversation_state: ConversationState,
        target: Optional[Union[PromptTarget, PromptChatTarget]] = None,
        max_turns: Optional[int] = None,
    ) -> None:
        """
        Process a prepended message and update the conversation state.
        This method handles the conversion of request pieces, sets conversation IDs,
        and attack identifiers, and processes each piece based on its role.

        Args:
            request (PromptRequestResponse): The request containing pieces to process.
            conversation_id (str): The ID of the conversation to update.
            conversation_state (ConversationState): The current state of the conversation.
            target (Optional[Union[PromptTarget, PromptChatTarget]]): The target to set system prompts on (if
                applicable).
            max_turns (Optional[int]): Maximum allowed turns for the conversation.

        Raises:
            ValueError: If the request is invalid or if a system prompt is provided but target doesn't support it.
        """
        # Validate the request before processing
        if not request or not request.request_pieces:
            return

        # Set the conversation ID and attack ID for each piece in the request
        save_to_memory = True
        for piece in request.request_pieces:
            piece.conversation_id = conversation_id
            piece.attack_identifier = self._attack_identifier
            piece.id = uuid.uuid4()

            # Process the piece based on its role
            self._process_piece(
                piece=piece,
                conversation_state=conversation_state,
                max_turns=max_turns,
                target=target,
            )

            if ConversationManager._should_exclude_piece_from_memory(piece=piece, max_turns=max_turns):
                # it is excluded, so we don't want to save it to memory
                save_to_memory = False

        # Add the request to memory if it was not a system piece
        if save_to_memory:
            self._memory.add_request_response_to_memory(request=request)

    def _process_piece(
        self,
        *,
        piece: PromptRequestPiece,
        conversation_state: ConversationState,
        max_turns: Optional[int] = None,
        target: Optional[Union[PromptTarget, PromptChatTarget]] = None,
    ) -> None:
        """
        Process a message piece based on its role and update conversation state.

        Args:
            piece (PromptRequestPiece): The piece to process.
            conversation_state (ConversationState): The current state of the conversation.
            max_turns (Optional[int]): Maximum allowed turns (for validation).
            target (Optional[Union[PromptTarget, PromptChatTarget]]): The target to set system prompts on.

        Raises:
            ValueError: If max_turns would be exceeded by this piece.
            ValueError: If a system prompt is provided but target doesn't support it.
        """

        # Check if multiturn
        is_multi_turn = max_turns is not None

        # Handle system prompts (both single-turn and multi-turn)
        if piece.role == "system":
            if target is None:
                raise ValueError("Target must be provided to handle system prompts")

            if not isinstance(target, PromptChatTarget):
                raise ValueError("Target must be a PromptChatTarget to set system prompts")

            # Set system prompt and exclude from memory
            self.set_system_prompt(
                target=target,
                conversation_id=piece.conversation_id,
                system_prompt=piece.converted_value,
                labels=piece.labels,
            )

        # Handle assistant messages (count turns for multi-turn only)
        elif piece.role == "assistant" and is_multi_turn:
            # Update turn count
            conversation_state.turn_count += 1

            # Validate against max_turns
            if max_turns and conversation_state.turn_count > max_turns:
                raise ValueError(
                    f"The number of turns in the prepended conversation ({conversation_state.turn_count-1}) is equal to"
                    + f" or exceeds the maximum number of turns ({max_turns}), which means the"
                    + " conversation will not be able to continue. Please reduce the number of turns in"
                    + " the prepended conversation or increase the maximum number of turns and try again."
                )

    @staticmethod
    def _should_exclude_piece_from_memory(*, piece: PromptRequestPiece, max_turns: Optional[int] = None) -> bool:
        # System pieces should always be excluded from memory because set_system_prompt function
        # is called on the target, which internally adds them to memory
        return piece.role == "system"

    async def _populate_conversation_state_async(
        self,
        *,
        prepended_conversation: List[PromptRequestResponse],
        last_message: PromptRequestPiece,
        conversation_state: ConversationState,
    ) -> None:
        """
        Extract conversation context from the last messages in prepended_conversation.

        This extracts:
        - Last user message for continuing conversations.
        - Scores for the last assistant message for evaluation.

        Args:
            prepended_conversation (List[PromptRequestResponse]): Complete conversation history.
            last_message (PromptRequestPiece): The last message in the history.
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
