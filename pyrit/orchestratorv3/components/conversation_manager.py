from dataclasses import dataclass, field
import logging
import uuid
from typing import Dict, List, Optional, Protocol, Union

from pyrit.memory import MemoryInterface, CentralMemory
from pyrit.models import ChatMessage, PromptRequestPiece, PromptRequestResponse
from pyrit.models.score import Score
from pyrit.prompt_target.common.prompt_chat_target import PromptChatTarget
from pyrit.prompt_target.common.prompt_target import PromptTarget

logger = logging.getLogger(__name__)

@dataclass
class ConversationState:
    """Container for conversation state data shared between orchestration components."""
    turn_count: int = 1
    last_user_message: str = ""
    last_assistant_message_scores: List[Score] = field(default_factory=list)

class ConversationManager:
    """
    Manages conversations for orchestrators.
    
    Responsible for:
    - Creating and managing conversations
    - Retrieving messages from conversations
    - Setting up system prompts
    - Preparing prepended conversations
    """
    
    def __init__(
        self,
        *,
        orchestrator_identifier: dict[str, str],
        memory: Optional[MemoryInterface] = None,
    ):
        """
        Initialize the conversation manager.
        
        Args:
            memory: The memory interface to use
            orchestrator_id: The ID of the orchestrator this manager belongs to
        """
        self._memory = memory or CentralMemory.get_memory_instance()
        self._orchestrator_identifier = orchestrator_identifier
    
    def get_conversation(self, conversation_id: str) -> List[PromptRequestResponse]:
        """
        Get all messages in a conversation.
        
        Args:
            conversation_id: The ID of the conversation.
            
        Returns:
            The list of messages in the conversation.
        """
        conversation = self._memory.get_conversation(conversation_id=conversation_id)
        return conversation
    
    def get_last_message(
        self, conversation_id: str, role: Optional[str] = None
    ) -> Optional[PromptRequestPiece]:
        """
        Get the last message in a conversation, optionally filtered by role.
        
        Args:
            conversation_id: The ID of the conversation.
            role: Optional role filter (e.g., 'user', 'assistant', 'system')
            
        Returns:
            The last message in the conversation, or None if no matching message exists.
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
    
    def add_system_prompt(
        self,
        target: Union[PromptTarget, PromptChatTarget],
        conversation_id: str,
        system_prompt: str,
        labels: Optional[Dict[str, str]] = None
    ) -> None:
        """
        Add a system prompt to a conversation.
        
        Args:
            target: The target to set the system prompt for
            conversation_id: The ID of the conversation
            system_prompt: The system prompt text
            labels: Optional memory labels to apply
        
        Raises:
            TypeError: If the target doesn't support system prompts
        """

        if not isinstance(target, PromptChatTarget):
            raise ValueError("Objective Target must be a PromptChatTarget to set system prompt.")
        
        target.set_system_prompt(
            system_prompt=system_prompt,
            conversation_id=conversation_id,
            orchestrator_identifier=self._orchestrator_identifier,
            labels=labels
        )
        
    def initialize_conversation_with_history(
        self,
        *,
        target: Optional[Union[PromptTarget, PromptChatTarget]] = None,
        max_turns: Optional[int] = None,
        conversation_id: str,
        history: List[PromptRequestResponse],
    ) -> ConversationState:
        """
        Initialize a conversation with history, preparing it for further interaction.
        
        This unified method handles both single-turn and multi-turn conversation preparation:
        - Updates conversation IDs on all messages
        - Sets system prompts if needed
        - Tracks turn count for multi-turn conversations
        - Extracts conversation state for continuation
        
        Args:
            target: Target to send system messages to (required for system messages)
            max_turns: Maximum allowed turns (required for multi-turn validation)
            conversation_id: ID to assign to the conversation
            history: Previous messages to initialize the conversation with
            
        Returns:
            ConversationState containing turn count and conversation context
            
        Raises:
            ValueError: If max_turns is provided and history exceeds this limit
            ValueError: If target doesn't support system prompts but system messages exist in history
            ValueError: If conversation_id is empty
        """
        if not conversation_id:
            raise ValueError("conversation_id cannot be empty")
            
        # Initialize conversation state
        state = ConversationState()
        logger.debug(f"Preparing conversation with ID: {conversation_id}")

        # Do not proceed if no history is provided
        if not history:
            logger.debug(f"No history provided for conversation initialization: {conversation_id}")
            return state
        
        # Determine if we should exclude the last message (if it's a user message in multi-turn context)
        last_message = history[-1].request_pieces[0]
        is_multi_turn = max_turns is not None
        should_exclude_last = is_multi_turn and last_message.role == "user"
        
        # Process all messages except potentially the last one
        for i, request in enumerate(history):
            # Skip the last message if it's a user message in multi-turn context
            if should_exclude_last and i == len(history) - 1:
                logger.debug("Skipping last user message (will be added by orchestrator)")
                continue

            self._process_prepended_message(
                request=request,
                conversation_id=conversation_id,
                conversation_state=state,
                target=target,
                max_turns=max_turns
            )

        # Extract state from the conversation history (only for multi-turn conversations)
        if is_multi_turn:
            self._extract_conversation_state(
                last_message=last_message,
                history=history,
                conversation_state=state,
            )

        return state
    
    def _process_prepended_message(
        self,
        *,
        request: PromptRequestResponse,
        conversation_id: str,
        conversation_state: ConversationState,
        target: Optional[Union[PromptTarget, PromptChatTarget]] = None,
        max_turns: Optional[int] = None,
    ) -> None:
        """
        Process a prepended message and add it to the conversation.
        
        Args:
            request: The request to process
            conversation_id: The ID of the conversation
            conversation_state: The current state of the conversation
            target: The target to set the system prompt for
        """
        # Validate the request before processing
        if not request or not request.request_pieces:
            return
        
        # Set the conversation ID and orchestrator ID for each piece in the request
        save_to_memory = True
        for piece in request.request_pieces:
            piece.conversation_id = conversation_id
            piece.orchestrator_identifier = self._orchestrator_identifier
            piece.id = uuid.uuid4()

            if (self._process_piece_and_check_if_excluded(
                piece=piece,
                conversation_state=conversation_state,
                max_turns=max_turns,
                target=target
            )):
                # it is excluded, so we don't want to save it to memory
                save_to_memory = False

        # Add the request to memory if it was not a system piece
        if save_to_memory:
            self._memory.add_request_response_to_memory(request=request)
                
    def _process_piece_and_check_if_excluded(
        self,
        *,
        piece: PromptRequestPiece,
        conversation_state: ConversationState,
        max_turns: Optional[int] = None,
        target: Optional[Union[PromptTarget, PromptChatTarget]] = None,
    ) -> bool:
        """
        Process a message piece based on its role and update conversation state.
        
        Args:
            piece: The piece to process
            conversation_state: The current state of the conversation
            max_turns: Maximum allowed turns (for validation)
            target: The target to set system prompts on
            
        Returns:
            bool: True if the piece should be EXCLUDED from memory storage, False otherwise
            
        Raises:
            ValueError: If max_turns would be exceeded by this piece
            ValueError: If a system prompt is provided but target doesn't support it
        """
        
        # Check if multiturn
        is_multi_turn = max_turns is not None

        # Basic checks
        # we don't exclude any pieces if we are not in multi-turn mode
        # and we only care about system and assistant roles
        if not is_multi_turn or piece.role not in ["system", "assistant"]:
            return False

        if piece.role == "system":
            if target is None:
                raise ValueError("Target must be provided to handle system prompts")
                
            if not isinstance(target, PromptChatTarget):
                raise ValueError("Target must be a PromptChatTarget to set system prompts")
                
            # Set system prompt and exclude from memory
            self.add_system_prompt(
                target=target,
                conversation_id=piece.conversation_id,
                system_prompt=piece.converted_value,
                labels=piece.labels,
            )
            return True  # Exclude from memory storage
        
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

        # By default, include in memory
        return False  # Don't exclude from memory
    
    def _extract_conversation_state(
        self,
        *,
        history: List[PromptRequestResponse],
        last_message: PromptRequestPiece,
        conversation_state: ConversationState,
    ) -> None:
        """
        Extract conversation context from the last messages in history.
        
        This extracts:
        - Last user message for continuing conversations
        - Scores for the last assistant message for evaluation
        
        Args:
            history: Complete conversation history
            last_message: The last message in the history
            conversation_state: State object to populate
        
        Raises:
            ValueError: If an assistant message doesn't have a preceding user message
        """
        if not history:
            return  # Nothing to extract from empty history
        
        # Extract the last user message and assistant message scores from the last message
        if last_message.role == "user":
            conversation_state.last_user_message = last_message.converted_value
            logger.debug(f"Extracted last user message: {conversation_state.last_user_message[:50]}...")

        elif last_message.role == "assistant":
            # Get scores for the last assistant message based off of the original id
            conversation_state.last_assistant_message_scores = self._memory.get_scores_by_prompt_ids(
                prompt_request_response_ids=[str(last_message.original_prompt_id)]
            )

            # Do not set last user message if there are no scores for the last assistant message
            if not conversation_state.last_assistant_message_scores:
                logger.debug("No scores found for last assistant message")
                return
            
            # Check assumption that there will be a user message preceding the assistant message
            if len(history) > 1 and history[-2].get_piece().role == "user":
                conversation_state.last_user_message = history[-2].get_value()
                logger.debug(f"Extracted preceding user message: {conversation_state.last_user_message[:50]}...")
            else:
                raise ValueError(
                    "There must be a user message preceding the assistant message in prepended conversations."
                )
