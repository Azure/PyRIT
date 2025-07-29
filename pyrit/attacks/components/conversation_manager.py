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
    Manages conversations for orchestrators, handling message history,
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
            orchestrator_identifier=self._attack_identifier,
            labels=labels,
        )

    async def update_conversation_state_async(
        self,
        *,
        conversation_id: str,
        target: Optional[Union[PromptTarget, PromptChatTarget]] = None,
        prepended_conversation: List[PromptRequestResponse],
        converter_configurations: Optional[List[PromptConverterConfiguration]] = None,
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
        the most recent user-utterance (so that an orchestrator can re-inject it as the "live" request),
        and extracts per-session counters such as the current turn index.

        Args:
            conversation_id (str): Unique identifier for the conversation to update or create.
            target (Optional[Union[PromptTarget, PromptChatTarget]]): The target to set system prompts on (if
                applicable).
            prepended_conversation (List[PromptRequestResponse]):
                List of messages to prepend to the conversation history.
            converter_configurations (Optional[List[PromptConverterConfiguration]]): List of configurations for
                converting prompt values.
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
                logger.debug("Skipping last user message (will be added by orchestrator)")
                continue

            # Apply converters if needed
            if converter_configurations:
                logger.debug(f"Converting request {i + 1}/{len(valid_requests)} in conversation {conversation_id}")
                # Convert the request values using the provided configurations
                await self._prompt_normalizer.convert_values(
                    request_response=request,
                    converter_configurations=converter_configurations,
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
        and orchestrator identifiers, and processes each piece based on its role.

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

        # Set the conversation ID and orchestrator ID for each piece in the request
        save_to_memory = True
        for piece in request.request_pieces:
            piece.conversation_id = conversation_id
            piece.orchestrator_identifier = self._attack_identifier
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

        # Basic checks
        # we don't exclude any pieces if we are not in multi-turn mode
        # and we only care about system and assistant roles
        if not is_multi_turn or piece.role not in ["system", "assistant"]:
            return

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

        # Handle assistant messages (count turns)
        elif piece.role == "assistant" and max_turns is not None:
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
        return max_turns is not None and piece.role == "system"

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
                self._memory.get_scores_by_prompt_ids(
                    prompt_request_response_ids=[str(last_message.original_prompt_id)]
                )
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
