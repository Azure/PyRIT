# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import uuid
from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Dict, Generic, List, Optional, Self, TypeVar, Union

from colorama import Fore, Style

from pyrit.common.display_response import display_image_response
from pyrit.memory.central_memory import CentralMemory
from pyrit.memory.memory_interface import MemoryInterface
from pyrit.models.literals import PromptDataType
from pyrit.models.prompt_request_piece import PromptRequestPiece
from pyrit.models.prompt_request_response import PromptRequestResponse
from pyrit.models.score import Score
from pyrit.prompt_converter.prompt_converter import PromptConverter

T_Context = TypeVar("T_Context", bound="AttackContext")
T_Result = TypeVar("T_Result", bound="AttackResult")


@dataclass
class AttackContext:
    """Base class for all attack contexts"""

    def duplicate(self) -> Self:
        """
        Create a deep copy of the context to avoid concurrency issues

        Returns:
            AttackContext: A deep copy of the context
        """
        return deepcopy(self)


@dataclass
class AttackResult:
    """Base class for all attack results"""

    orchestrator_identifier: dict[str, str]


class BacktrackingStrategy(ABC, Generic[T_Context]):
    """Base class for backtracking strategies"""

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name of the backtracking strategy"""
        pass

    @abstractmethod
    async def should_backtrack(self, context: T_Context, response: PromptRequestPiece, score: Optional[Score]) -> bool:
        """Determine if backtracking should be applied"""
        pass

    @abstractmethod
    async def apply_backtracking(self, context: T_Context) -> None:
        """Apply backtracking to the context"""
        pass


@dataclass
class ConversationSession:
    """Session for conversations"""

    conversation_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    adversarial_chat_conversation_id: str = field(default_factory=lambda: str(uuid.uuid4()))


@dataclass
class MultiTurnAttackContext(AttackContext):
    """Context for multi-turn attacks"""

    session: ConversationSession = field(default_factory=lambda: ConversationSession())
    objective: Optional[str] = None
    max_turns: int = 5
    achieved_objective: bool = False
    executed_turns: int = 0
    last_response: Optional[PromptRequestPiece] = None
    last_score: Optional[Score] = None
    custom_prompt: Optional[str] = None
    prompt_converters: List[PromptConverter] = field(default_factory=list)
    prepended_conversation: List[PromptRequestResponse] = field(default_factory=list)
    memory_labels: Optional[Dict[str, str]] = None


@dataclass
class SingleTurnAttackContext(AttackContext):
    """Context for single-turn attacks"""

    batch_size: int = 1
    prompts: List[str] = field(default_factory=list)
    prompt_data_type: PromptDataType = "text"
    prepended_conversation: List[PromptRequestResponse] = field(default_factory=list)
    memory_labels: Optional[Dict[str, str]] = None
    metadata: Optional[dict[str, Union[str, int]]] = None


@dataclass
class SingleTurnAttackResult(AttackResult):
    """Result of a single-turn attack"""

    prompt_list: List[PromptRequestResponse] = field(default_factory=list)

    async def print_conversations(self) -> None:
        """Print the conversations associated with this attack result

        Args:
            include_scores: Whether to include scoring information
        """
        memory = CentralMemory.get_memory_instance()
        messages = memory.get_prompt_request_pieces(orchestrator_id=self.orchestrator_identifier["id"])

        if not messages or len(messages) == 0:
            print("No conversations found for this orchestrator ID")
            return

        last_conversation_id = None

        for message in messages:
            # Print conversation ID header when it changes
            if message.conversation_id != last_conversation_id:
                if last_conversation_id is not None:
                    # Add a separator between conversations
                    print(f"\n{'-'*50}\n")
                print(f"{Style.BRIGHT}{Fore.CYAN}Conversation ID: {message.conversation_id}{Style.RESET_ALL}")
                last_conversation_id = message.conversation_id

            # Print message based on role
            if message.role in ["user", "system"]:
                print(f"{Style.BRIGHT}{Fore.BLUE}{message.role}: {message.converted_value}")
            else:
                print(f"{Style.NORMAL}{Fore.YELLOW}{message.role}: {message.converted_value}")
                # Display images if any
                await display_image_response(message)

            # Print scores
            for score in message.scores:
                print(f"{Style.RESET_ALL}score: {score} : {score.score_rationale}")


@dataclass
class MultiTurnAttackResult(AttackResult):
    """Result of a multi-turn attack"""

    conversation_id: str
    objective: Optional[str] = None
    achieved_objective: bool = False
    executed_turns: int = 0
    last_response: Optional[PromptRequestPiece] = None
    last_score: Optional[Score] = None
    memory_labels: Optional[Dict[str, str]] = None

    async def print_conversation(self, *, include_scores: bool = True) -> None:
        """
        Print the conversation between the target and the adversarial chat

        Args:
            include_scores: Whether to include scoring information
        """
        memory = CentralMemory.get_memory_instance()
        target_messages = memory.get_conversation(conversation_id=self.conversation_id)

        if not target_messages or len(target_messages) == 0:
            print("No conversation with the target")
            return

        # Print objective status
        self._print_objective_status()

        # Print each message and its pieces
        for message in target_messages:
            await self._print_message(message=message, memory=memory, include_scores=include_scores)

    def _print_objective_status(self) -> None:
        """Print whether the objective was achieved"""
        if self.achieved_objective:
            print(
                f"{Style.BRIGHT}{Fore.RED}The multi-turn orchestrator has completed the conversation and achieved "
                f"the objective: {self.objective}"
            )
        else:
            print(
                f"{Style.BRIGHT}{Fore.RED}The multi-turn orchestrator has not achieved the objective: "
                f"{self.objective}"
            )

    async def _print_message(
        self, *, message: PromptRequestResponse, memory: MemoryInterface, include_scores: bool
    ) -> None:
        """
        Print a single message with all its pieces

        Args:
            message: The message to print
            memory: Memory interface to retrieve scores
            include_scores: Whether to include scoring information
        """
        for piece in message.request_pieces:
            # Print user messages
            if piece.role == "user":
                print(f"{Style.BRIGHT}{Fore.BLUE}{piece.role}:")
                if piece.converted_value != piece.original_value:
                    print(f"Original value: {piece.original_value}")
                print(f"Converted value: {piece.converted_value}")
            # Print assistant/system messages
            else:
                print(f"{Style.NORMAL}{Fore.YELLOW}{piece.role}: {piece.converted_value}")

            # Display images if any
            await display_image_response(piece)

            # Optionally print scores
            if include_scores:
                await self._print_scores_for_piece(piece=piece, memory=memory)

    async def _print_scores_for_piece(self, *, piece: PromptRequestPiece, memory: MemoryInterface) -> None:
        """
        Print scores associated with a message piece

        Args:
            piece: The message piece to print scores for
            memory: Memory interface to retrieve scores
        """
        scores = memory.get_scores_by_prompt_ids(prompt_request_response_ids=[str(piece.id)])
        if scores and len(scores) > 0:
            for score in scores:
                print(f"{Style.RESET_ALL}score: {score} : {score.score_rationale}")
