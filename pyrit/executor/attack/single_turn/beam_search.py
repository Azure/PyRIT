# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import asyncio
import copy
import logging
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional, Type

from pyrit.common.apply_defaults import REQUIRED_VALUE, apply_defaults
from pyrit.common.utils import warn_if_set
from pyrit.executor.attack.component import ConversationManager, PrependedConversationConfig
from pyrit.executor.attack.core import AttackConverterConfig, AttackScoringConfig
from pyrit.executor.attack.core.attack_parameters import AttackParameters, AttackParamsT
from pyrit.executor.attack.single_turn.single_turn_attack_strategy import (
    SingleTurnAttackContext,
    SingleTurnAttackStrategy,
)
from pyrit.models import (
    AttackOutcome,
    AttackResult,
    Message,
    Score,
)
from pyrit.prompt_normalizer import PromptNormalizer
from pyrit.prompt_target import OpenAIResponseTarget
from pyrit.score import Scorer

logger = logging.getLogger(__name__)


def _print_message(message: Message) -> None:
    for piece in message.message_pieces:
        if piece.role == "user":
            # User message header
            print()
            print("─" * 10)
            print("USER")

            # Handle converted values
            if piece.converted_value != piece.original_value:
                print(f"Original:")
                print(piece.original_value)
                print()
                print(f"Converted:")
                print(piece.converted_value)
            else:
                print(piece.converted_value)
        elif piece.role == "system":
            # System message header (not counted as a turn)
            print()
            print("─" * 10)
            print("SYSTEM")

            print(piece.converted_value)
        else:
            if piece.original_value_data_type != "reasoning":
                # Assistant message header
                print()
                print("─" * 10)
                print(f"{piece.role.upper()}")

                print(piece.converted_value)


@dataclass
class Beam:
    """Represents a beam in the beam search attack strategy."""

    id: str
    text: str
    score: float
    objective_score: Optional[Score] = None
    message: Message | None = None


class BeamPruner(ABC):
    """Abstract base class for beam pruners in beam search attacks."""

    @abstractmethod
    def prune(self, beams: list[Beam]) -> list[Beam]:
        """
        Remove less promising beams from the list, replacing them with modified copies of the better beams.

        Args:
            beams (list[Beam]): The current list of beams.

        Returns:
            list[Beam]: The updated list of beams.
        """
        pass


class TopKBeamPruner(BeamPruner):
    """Beam pruner that retains the top-k beams and modifies them to create new beams."""

    def __init__(self, k: int, drop_chars: int):
        """
        Initialize the TopKBeamPruner.

        Args:
            k (int): The number of top beams to retain.
            drop_chars (int): The number of characters to drop from the end of the retained beams
                to create new beams.
        """
        self.k = k
        self.drop_chars = drop_chars

    def prune(self, beams: list[Beam]) -> list[Beam]:
        """
        Prune the beams to retain the top-k and create new beams by modifying them.

        Args:
            beams (list[Beam]): The current list of beams.

        Returns:
            list[Beam]: The updated list of beams.
        """
        # Sort beams by score in descending order and select top k
        sorted_beams = sorted(beams, key=lambda b: b.score, reverse=True)

        new_beams = list(reversed(sorted_beams[: self.k]))
        for i in range(len(beams) - len(new_beams)):
            nxt = copy.deepcopy(new_beams[i % self.k])
            if len(nxt.text) > self.drop_chars:
                nxt.text = nxt.text[: -self.drop_chars]
            new_beams.append(nxt)
        assert len(beams) == len(new_beams)
        return new_beams


class BeamSearchAttack(SingleTurnAttackStrategy):
    @apply_defaults
    def __init__(
        self,
        *,
        objective_target: OpenAIResponseTarget = REQUIRED_VALUE,  # type: ignore[assignment]
        beam_pruner: BeamPruner = REQUIRED_VALUE,  # type: ignore[assignment]
        attack_converter_config: Optional[AttackConverterConfig] = None,
        attack_scoring_config: Optional[AttackScoringConfig] = None,
        prompt_normalizer: Optional[PromptNormalizer] = None,
        params_type: Type[AttackParamsT] = AttackParameters,  # type: ignore[assignment]
        prepended_conversation_config: Optional[PrependedConversationConfig] = None,
        num_beams: int = 2,
        max_iterations: int = 4,
        num_chars_per_step: int = 100,
    ) -> None:
        """
        Initialize the prompt injection attack strategy.

        Args:
            objective_target (OpenAIResponseTarget): The target system to attack.
            beam_pruner (BeamPruner): The beam pruner to use during the search.
            attack_converter_config (Optional[AttackConverterConfig]): Configuration for prompt converters.
            attack_scoring_config (Optional[AttackScoringConfig]): Configuration for scoring components.
            prompt_normalizer (Optional[PromptNormalizer]): The prompt normalizer to use.
            params_type (Type[AttackParamsT]): The type of attack parameters to use.
            prepended_conversation_config (Optional[PrependedConversationConfig]): Configuration for prepended conversation.
            num_beams (int): The number of beams to use in the search.
            max_iterations (int): The maximum number of iterations to perform.
            num_chars_per_step (int): The number of characters to generate per step.
        """
        # Initialize base class
        super().__init__(
            objective_target=objective_target,
            logger=logger,
            context_type=SingleTurnAttackContext,
            params_type=params_type,
        )

        # Initialize the converter configuration
        attack_converter_config = attack_converter_config or AttackConverterConfig()
        self._request_converters = attack_converter_config.request_converters
        self._response_converters = attack_converter_config.response_converters

        # Initialize scoring configuration
        attack_scoring_config = attack_scoring_config or AttackScoringConfig()

        # Check for unused optional parameters and warn if they are set
        warn_if_set(config=attack_scoring_config, unused_fields=["refusal_scorer"], log=logger)

        self._auxiliary_scorers = attack_scoring_config.auxiliary_scorers
        self._objective_scorer = attack_scoring_config.objective_scorer

        # Skip criteria could be set directly in the injected prompt normalizer
        self._prompt_normalizer = prompt_normalizer or PromptNormalizer()
        self._conversation_manager = ConversationManager(
            attack_identifier=self.get_identifier(),
            prompt_normalizer=self._prompt_normalizer,
        )

        self._beam_pruner = beam_pruner

        self._num_beams = num_beams
        self._max_iterations = max_iterations
        self._num_chars_per_step = num_chars_per_step

        # Store the prepended conversation configuration
        self._prepended_conversation_config = prepended_conversation_config

    def _validate_context(self, *, context: SingleTurnAttackContext) -> None:
        """
        Validate the context before executing the attack.

        Args:
            context (SingleTurnAttackContext): The attack context containing parameters and objective.

        Raises:
            ValueError: If the context is invalid.
        """
        if not context.objective or context.objective.isspace():
            raise ValueError("Attack objective must be provided and non-empty in the context")

    async def _setup_async(self, *, context: SingleTurnAttackContext) -> None:
        """
        Set up the attack by preparing conversation context.

        Args:
            context (SingleTurnAttackContext): The attack context containing attack parameters.
        """
        self._start_context = copy.deepcopy(context)

        # Ensure the context has a conversation ID
        context.conversation_id = str(uuid.uuid4())

        # Initialize context with prepended conversation and merged labels
        await self._conversation_manager.initialize_context_async(
            context=context,
            target=self._objective_target,
            conversation_id=context.conversation_id,
            request_converters=self._request_converters,
            prepended_conversation_config=self._prepended_conversation_config,
            memory_labels=self._memory_labels,
        )

    async def _perform_async(self, *, context: SingleTurnAttackContext[Any]) -> AttackResult:
        """
        Perform the attack.

        Args:
            context: The attack context with objective and parameters.

        Returns:
            AttackResult containing the outcome of the attack.
        """
        # Log the attack configuration
        self._logger.info(f"Starting {self.__class__.__name__} with objective: {context.objective}")

        beams = [Beam(id=context.conversation_id, text="", score=0.0) for _ in range(self._num_beams)]

        for step in range(self._max_iterations):
            print(f"Starting iteration {step}")

            # Prune beams at the top of the loop for simplicity
            beams = self._beam_pruner.prune(beams)

            async with asyncio.TaskGroup() as tg:
                tasks = [tg.create_task(self._propagate_beam(beam=beam)) for beam in beams]
                await asyncio.gather(*tasks)

            for i, beam in enumerate(beams):
                print(f"Beam {i} text after iteration {step}: {beam.text}")

            print("Scoring beams")
            async with asyncio.TaskGroup() as tg:
                tasks = [tg.create_task(self._score_beam(beam=beam, context=context)) for beam in beams]
                scores = await asyncio.gather(*tasks)

            for i, beam in enumerate(beams):
                print(f"Beam {i} score: {beam.score}")

        # Sort the list of beams
        beams = sorted(beams, key=lambda b: b.score, reverse=True)

        outcome, outcome_reason = self._determine_attack_outcome(beam=beams[0])

        result = AttackResult(
            conversation_id=beams[0].id,
            objective=context.objective,
            attack_identifier=self.get_identifier(),
            last_response=beams[0].message.message_pieces[0] if beams[0].message else None,
            last_score=beams[0].objective_score,
            outcome=outcome,
            outcome_reason=outcome_reason,
            executed_turns=1,
        )

        return result

    async def _propagate_beam(self, *, beam: Beam):
        # print(f"Propagating beam with text: {beam.text}")
        target = self._get_target_for_beam(beam)

        current_context = copy.deepcopy(self._start_context)
        await self._setup_async(context=current_context)

        message = self._get_message(current_context)
        beam.id = current_context.conversation_id

        try:
            model_response = await self._prompt_normalizer.send_prompt_async(
                message=message,
                target=target,
                conversation_id=current_context.conversation_id,
                request_converter_configurations=self._request_converters,
                response_converter_configurations=self._response_converters,
                labels=current_context.memory_labels,  # combined with strategy labels at _setup()
                attack_identifier=self.get_identifier(),
            )

            # _print_message(model_response)
            assert len(model_response.message_pieces) == 2, "Expected exactly two message pieces in the response"
            model_response.message_pieces = model_response.message_pieces[1:]
            model_response.message_pieces[0].role = "assistant"
            beam.text = model_response.message_pieces[0].converted_value
            beam.message = model_response
            # print(f"Updated beam text: {beam.text}")
        except Exception as e:
            # Just log the error and skip the update
            logger.warning(f"Error propagating beam: {e}")

    async def _score_beam(self, *, beam: Beam, context: SingleTurnAttackContext[Any]):
        assert beam.message is not None, "Beam message must be set before scoring"
        scoring_results = await Scorer.score_response_async(
            response=beam.message,
            objective_scorer=self._objective_scorer,
            auxiliary_scorers=self._auxiliary_scorers,
            role_filter="assistant",
            objective=context.objective,
        )

        aux_scores = scoring_results["auxiliary_scores"]
        beam.score = 0.0
        for s in aux_scores:
            # print(f"Auxiliary score: {s}")
            # print(f"{s.get_value()=}")
            beam.score += s.get_value()

        objective_scores = scoring_results["objective_scores"]
        if objective_scores:
            beam.objective_score = objective_scores[0]

    def _get_target_for_beam(self, beam: Beam) -> OpenAIResponseTarget:
        """
        Create a OpenAIResponseTarget for the given beam by duplicating the base target.

        Args:
            beam (Beam): The beam for which to create the target.

        Returns:
            OpenAIResponseTarget: A new target configured for the beam.

        Raises:
            ValueError: If the objective target is not an OpenAIResponseTarget.
        """
        grammar_template = """
start: PREFIX CONTINUATION
PREFIX: "{prefix}"
CONTINUATION: /.{{0,{n_chars}}}/
"""
        if not(isinstance(self._objective_target, OpenAIResponseTarget)):
            raise ValueError("Objective target must be an OpenAIResponseTarget")
        lark_grammar = grammar_template.format(prefix=beam.text.replace('"', '\\"'), n_chars=self._num_chars_per_step)

        grammar_tool = {
            "type": "custom",
            "name": "ContinuationGrammar",
            "description": "Forces continuation of the given prefix.",
            "format": {
                "type": "grammar",
                "syntax": "lark",
                "definition": lark_grammar,
            },
        }

        reasoning = {"effort": "minimal"}

        ebp = {
            "reasoning": reasoning,
            "tools": [grammar_tool],
            "tool_choice": "required",
        }

        target = self._objective_target.fresh_instance()
        target._extra_body_parameters = ebp
        target._grammar_name = grammar_tool["name"]

        return target

    def _get_message(self, context: SingleTurnAttackContext[Any]) -> Message:
        """
        Prepare the message for the attack.

        If a message is provided in the context, it will be used directly.
        Otherwise, creates a new Message from the objective as a text prompt.

        Args:
            context (SingleTurnAttackContext): The attack context containing the objective
                and optionally a pre-configured message template.

        Returns:
            Message: The message to be used in the attack.
        """
        if context.next_message:
            # Deep copy the message to preserve all fields, then assign new IDs
            return context.next_message.duplicate_message()

        return Message.from_prompt(prompt=context.objective, role="user")


    def _determine_attack_outcome(
        self, *, beam: Beam
    ) -> tuple[AttackOutcome, Optional[str]]:
        """
        Determine the outcome of the attack based on the response and score.

        Args:
           beam (Beam): The beam containing the response and score.

        Returns:
            tuple[AttackOutcome, Optional[str]]: A tuple of (outcome, outcome_reason).
        """
        if not self._objective_scorer:
            # No scorer means we can't determine success/failure
            return AttackOutcome.UNDETERMINED, "No objective scorer configured"

        if beam.objective_score and beam.objective_score.get_value():
            # We have a positive score, so it's a success
            return AttackOutcome.SUCCESS, "Objective achieved according to scorer"

        # No response at all (all attempts filtered/failed)
        return AttackOutcome.FAILURE, "All attempts were filtered or failed to get a response"

    async def _teardown_async(self, *, context: SingleTurnAttackContext) -> None:
        """Clean up after attack execution."""
        # Nothing to be done here, no-op
        pass
