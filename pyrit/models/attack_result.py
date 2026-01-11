# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

import warnings
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, List, MutableSequence, Optional, TypeVar

from pyrit.models.conversation_reference import ConversationReference, ConversationType
from pyrit.models.message_piece import MessagePiece
from pyrit.models.score import Score
from pyrit.models.strategy_result import StrategyResult

if TYPE_CHECKING:
    from pyrit.models.message import Message

AttackResultT = TypeVar("AttackResultT", bound="AttackResult")


class AttackOutcome(Enum):
    """
    Enum representing the possible outcomes of an attack.
    """

    # The attack was successful in achieving its objective
    SUCCESS = "success"

    # The attack failed to achieve its objective
    FAILURE = "failure"

    # The outcome of the attack is unknown or could not be determined
    UNDETERMINED = "undetermined"


class AttackResult(StrategyResult):
    """
    Base class for all attack results.

    Contains identity information, scoring, metadata moved from per-message storage,
    and methods to retrieve conversation history.
    """

    def __init__(
        self,
        *,
        conversation_id: str,
        objective: str,
        attack_identifier: dict[str, str],
        targeted_harm_categories: Optional[List[str]] = None,
        request_converter_identifiers: Optional[List[Dict[str, str]]] = None,
        objective_target_identifier: Optional[Dict[str, str]] = None,
        labels: Optional[Dict[str, str]] = None,
        automated_objective_score: Optional[Score] = None,
        human_objective_score: Optional[Score] = None,
        auxiliary_score_ids: Optional[List[str]] = None,
        executed_turns: int = 0,
        execution_time_ms: int = 0,
        outcome: Optional[AttackOutcome] = None,
        outcome_reason: Optional[str] = None,
        related_conversations: Optional[set[ConversationReference]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Initialize an AttackResult.

        Args:
            conversation_id: Unique identifier of the conversation that produced this result.
            objective: Natural-language description of the attacker's objective.
            attack_identifier: Identifier of the attack (e.g., name, module).
            targeted_harm_categories: Harm categories associated with this attack.
            request_converter_identifiers: Converter identifiers used during the attack.
            objective_target_identifier: Target identifier for the attack.
            labels: Labels associated with this attack.
            automated_objective_score: The automated objective score (must be true_false type).
            human_objective_score: The human objective score (must be true_false type).
            auxiliary_score_ids: IDs of additional scores providing auxiliary information.
            executed_turns: Total number of turns that were executed.
            execution_time_ms: Total execution time of the attack in milliseconds.
            outcome: The outcome of the attack. If None, derived from objective_score.
            outcome_reason: Optional reason for the outcome.
            related_conversations: Set of related conversation references.
            metadata: Arbitrary metadata dictionary.
        """
        # Identity
        self.conversation_id = conversation_id
        self.objective = objective
        self.attack_identifier = attack_identifier

        # Metadata moved from MessagePiece (stored once per attack, not per message)
        self.targeted_harm_categories = targeted_harm_categories
        self.request_converter_identifiers = request_converter_identifiers
        self.objective_target_identifier = objective_target_identifier
        self.labels = labels

        # Private backing fields for scores (use property setters for validation)
        self._automated_objective_score: Optional[Score] = None
        self._human_objective_score: Optional[Score] = None

        # Use setters for validation
        self.automated_objective_score = automated_objective_score
        self.human_objective_score = human_objective_score

        # Auxiliary scores
        self.auxiliary_score_ids = auxiliary_score_ids if auxiliary_score_ids is not None else []

        # Metrics
        self.executed_turns = executed_turns
        self.execution_time_ms = execution_time_ms

        # Outcome - derive from objective_score if not provided
        if outcome is not None:
            self.outcome = outcome
        elif self.objective_score is not None:
            self.outcome = AttackOutcome.SUCCESS if self.objective_score.get_value() else AttackOutcome.FAILURE
        else:
            self.outcome = AttackOutcome.UNDETERMINED

        self.outcome_reason = outcome_reason

        # Related conversations
        self.related_conversations = related_conversations if related_conversations is not None else set()

        # Metadata
        self.metadata = metadata if metadata is not None else {}

    @property
    def objective_score(self) -> Optional[Score]:
        """
        Get the effective objective score for this attack.

        If a human objective score has been set, it takes precedence over the automated score.

        Returns:
            Optional[Score]: The human objective score if set, otherwise the automated objective score.
        """
        if self._human_objective_score is not None:
            return self._human_objective_score
        return self._automated_objective_score

    @property
    def automated_objective_score(self) -> Optional[Score]:
        """Get the automated objective score."""
        return self._automated_objective_score

    @automated_objective_score.setter
    def automated_objective_score(self, value: Optional[Score]) -> None:
        """
        Set the automated objective score.

        Args:
            value: The score to set. Must be a true_false type score if provided.

        Raises:
            ValueError: If the score is not a true_false type.
        """
        if value is not None and value.score_type != "true_false":
            raise ValueError("automated_objective_score must be a true_false type score")
        self._automated_objective_score = value

    @property
    def human_objective_score(self) -> Optional[Score]:
        """Get the human objective score."""
        return self._human_objective_score

    @human_objective_score.setter
    def human_objective_score(self, value: Optional[Score]) -> None:
        """
        Set the human objective score, which overrides the automated_objective_score.

        Args:
            value: The score to set. Must be a true_false type score if provided.

        Raises:
            ValueError: If the score is not a true_false type.
        """
        if value is not None and value.score_type != "true_false":
            raise ValueError("human_objective_score must be a true_false type score")
        self._human_objective_score = value

    @property
    def last_response(self) -> Optional[MessagePiece]:
        """
        Deprecated: Get the last response from the conversation.

        This property is deprecated and will be removed in 0.13.0.
        Use get_conversation() instead to retrieve conversation messages.
        """
        warnings.warn(
            "AttackResult.last_response is deprecated and will be removed in 0.13.0. "
            "Use get_conversation() to retrieve conversation messages.",
            DeprecationWarning,
            stacklevel=2,
        )
        # Check if a value was explicitly set via the deprecated setter
        if hasattr(self, "_deprecated_last_response") and self._deprecated_last_response is not None:
            return self._deprecated_last_response
        conversation = self.get_conversation()
        if conversation:
            return conversation[-1].get_piece() if hasattr(conversation[-1], "get_piece") else None
        return None

    @last_response.setter
    def last_response(self, value: Optional[MessagePiece]) -> None:
        """
        Deprecated: Set the last response.

        This property is deprecated and will be removed in 0.13.0.
        """
        warnings.warn(
            "AttackResult.last_response is deprecated and will be removed in 0.13.0. "
            "Use get_conversation() to retrieve conversation messages.",
            DeprecationWarning,
            stacklevel=2,
        )
        self._deprecated_last_response = value

    @property
    def last_score(self) -> Optional[Score]:
        """
        Deprecated: Get the last score.

        This property is deprecated and will be removed in 0.13.0.
        Use objective_score instead.
        """
        warnings.warn(
            "AttackResult.last_score is deprecated and will be removed in 0.13.0. Use objective_score instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.objective_score

    @last_score.setter
    def last_score(self, value: Optional[Score]) -> None:
        """
        Deprecated: Set the last score.

        This property is deprecated and will be removed in 0.13.0.
        Use automated_objective_score instead.

        Args:
            value: The score to set. Must be a true_false type score if provided.

        Raises:
            ValueError: If the score is not a true_false type.
        """
        warnings.warn(
            "AttackResult.last_score is deprecated and will be removed in 0.13.0. "
            "Use automated_objective_score instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        # Use the property setter to enforce validation
        self.automated_objective_score = value

    def get_conversation(self) -> MutableSequence["Message"]:
        """
        Retrieve the full conversation associated with this attack result.

        This method fetches all messages from memory using the conversation_id.

        Returns:
            MutableSequence[Message]: The list of messages in the conversation.
        """
        from pyrit.memory import CentralMemory

        memory = CentralMemory.get_memory_instance()
        return memory.get_conversation(conversation_id=self.conversation_id)

    def get_auxiliary_scores(self) -> List[Score]:
        """
        Retrieve the auxiliary scores associated with this attack result.

        This method fetches all scores from memory using the auxiliary_score_ids.

        Returns:
            List[Score]: The list of auxiliary scores.
        """
        if not self.auxiliary_score_ids:
            return []

        from pyrit.memory import CentralMemory

        memory = CentralMemory.get_memory_instance()
        return list(memory.get_prompt_scores(prompt_ids=self.auxiliary_score_ids))

    def get_conversation_ids_by_type(self, conversation_type: ConversationType) -> List[ConversationReference]:
        """
        Return all related conversations of the requested type.

        Args:
            conversation_type: The type of conversation to filter by.

        Returns:
            List[ConversationReference]: A list of related conversations matching the specified type.
        """
        return [ref for ref in self.related_conversations if ref.conversation_type == conversation_type]

    def __str__(self) -> str:
        outcome_value = self.outcome.value if self.outcome else "unknown"
        return f"AttackResult: {self.conversation_id}: {outcome_value}: {self.objective[:50]}..."

    def __repr__(self) -> str:
        return (
            f"AttackResult(conversation_id={self.conversation_id!r}, "
            f"objective={self.objective[:30]!r}..., "
            f"outcome={self.outcome}, "
            f"objective_score={self.objective_score})"
        )
