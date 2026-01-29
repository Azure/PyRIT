# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import asyncio
import copy
import logging
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional

from pyrit.common.apply_defaults import REQUIRED_VALUE, apply_defaults
from pyrit.common.utils import combine_dict, warn_if_set
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
    ConversationReference,
    ConversationType,
    Message,
    Score,
    SeedGroup,
    SeedPrompt,
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
    id: str
    text: str
    score: float
    context: SingleTurnAttackContext | None = None
    message: Message | None = None


class BeamSearchAttack(SingleTurnAttackStrategy):
    @apply_defaults
    def __init__(
        self,
        *,
        objective_target: PromptTarget = REQUIRED_VALUE,  # type: ignore[assignment]
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
            attack_converter_config (Optional[AttackConverterConfig]): Configuration for prompt converters.
            attack_scoring_config (Optional[AttackScoringConfig]): Configuration for scoring components.

        Raises:
            ValueError: If the objective scorer is not a true/false scorer.
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

        print(f"Context: {context}")

        # Execute with retries
        response = None
        score = None

        # Prepare a fresh message for each attempt to avoid duplicate ID errors in database
        message = self._get_message(context)

        beams = [Beam(id=context.conversation_id, text="", score=0.0) for _ in range(self._num_beams)]

        for step in range(self._max_iterations):
            print(f"Starting iteration {step}")
            async with asyncio.TaskGroup() as tg:
                tasks = [
                    tg.create_task(
                        self._propagate_beam(
                            beam=beam, first_call=step == 0, prompt_group=prompt_group, context=context
                        )
                    )
                    for beam in beams
                ]
                await asyncio.gather(*tasks)

            for i, beam in enumerate(beams):
                print(f"Beam {i} text after iteration {step}: {beam.text}")

            print("Scoring beams")
            async with asyncio.TaskGroup() as tg:
                tasks = [tg.create_task(self._score_beam(beam=beam)) for beam in beams]
                scores = await asyncio.gather(*tasks)

            for i, beam in enumerate(beams):
                print(f"Beam {i} score: {beam.score}")

            for s in scores:
                print(f"Score: {s}")

        result = AttackResult(
            conversation_id=context.conversation_id,
            objective=context.objective,
            attack_identifier=self.get_identifier(),
            last_response=response.get_piece() if response else None,
            last_score=score,
            related_conversations=context.related_conversations,
            # outcome=outcome,
            # outcome_reason=outcome_reason,
            executed_turns=1,
        )

        return result

    async def _propagate_beam(
        self, *, beam: Beam, first_call: bool, message: Message, context: SingleTurnAttackContext[Any]
    ):
        print(f"Propagating beam with text: {beam.text}")
        target = self._get_target_for_beam(beam)

        if first_call:
            new_conversation_id = target._memory.duplicate_conversation(conversation_id=context.conversation_id)
        else:
            new_conversation_id = target._memory.duplicate_conversation(conversation_id=context.conversation_id)

        new_context = copy.deepcopy(context)
        new_context.conversation_id = new_conversation_id
        beam.id = new_conversation_id
        beam.context = new_context

        model_response = await self._prompt_normalizer.send_prompt_async(
            message=message,
            target=target,
            conversation_id=new_context.conversation_id,
            request_converter_configurations=self._request_converters,
            response_converter_configurations=self._response_converters,
            labels=context.memory_labels,  # combined with strategy labels at _setup()
            attack_identifier=self.get_identifier(),
        )

        # _print_message(model_response)
        assert len(model_response.message_pieces) == 2, "Expected exactly two message pieces in the response"
        model_response.message_pieces = model_response.message_pieces[1:]
        model_response.message_pieces[0].role = "assistant"
        beam.text = model_response.message_pieces[0].converted_value
        beam.message = model_response
        print(f"Updated beam text: {beam.text}")

    async def _score_beam(self, *, beam: Beam) -> Optional[Score]:
        assert beam.message is not None, "Beam message must be set before scoring"
        assert beam.context is not None, "Beam context must be set before scoring"
        scoring_results = await Scorer.score_response_async(
            response=beam.message,
            objective_scorer=self._objective_scorer,
            auxiliary_scorers=self._auxiliary_scorers,
            role_filter="assistant",
            objective=beam.context.objective,
        )

        aux_scores = scoring_results["auxiliary_scores"]
        beam.score = 0.0
        for s in aux_scores:
            print(f"Auxiliary score: {s}")
            print(f"{s.get_value()=}")
            beam.score += s.get_value()

        objective_scores = scoring_results["objective_scores"]
        if not objective_scores:
            return None
        return objective_scores[0]

    def _get_target_for_beam(self, beam: Beam) -> OpenAIResponseTarget:
        """
        Create a OpenAIResponseTarget for the given beam by duplicating the base target.

        Args:
            beam (Beam): The beam for which to create the target.
        """
        grammar_template = """
start: PREFIX CONTINUATION
PREFIX: "{prefix}"
CONTINUATION: /.{{0,{n_chars}}}/
"""

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

        return SeedGroup(prompts=[SeedPrompt(value=context.objective, data_type="text")])

    async def _teardown_async(self, *, context: SingleTurnAttackContext) -> None:
        """Clean up after attack execution."""
        # Nothing to be done here, no-op
        pass
