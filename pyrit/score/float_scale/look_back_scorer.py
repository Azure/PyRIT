# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import logging
import re
from pathlib import Path
from typing import ClassVar, Optional

import yaml

from pyrit.common.path import SCORER_LIKERT_PATH, SCORER_SCALES_PATH
from pyrit.models import MessagePiece, Score, SeedPrompt, UnvalidatedScore
from pyrit.prompt_target import PromptChatTarget
from pyrit.score.float_scale.float_scale_scorer import FloatScaleScorer
from pyrit.score.scorer_prompt_validator import ScorerPromptValidator

logger = logging.getLogger(__name__)


class LookBackScorer(FloatScaleScorer):
    """
    Create a score from analyzing the entire conversation and adds them to the database.
    """

    version: ClassVar[int] = 1
    _default_validator: ScorerPromptValidator = ScorerPromptValidator()

    def __init__(
        self,
        chat_target: PromptChatTarget,
        *,
        exclude_instruction_prompts: bool,
        validator: Optional[ScorerPromptValidator] = None,
    ) -> None:
        """
        Initialize the LookBackScorer.

        Args:
            chat_target (PromptChatTarget): The chat target to use for scoring.
            exclude_instruction_prompts (bool): If True, excludes instruction prompts from the conversation.
                Must be passed as a keyword argument.
            validator: Optional validator for the scorer.
        """
        super().__init__(validator=validator or self._default_validator)
        self._prompt_target = chat_target

        self.exclude_instruction_prompts = exclude_instruction_prompts

        behavior_change_prompt_path = Path(SCORER_SCALES_PATH, "behavior_change_system_prompt.yaml").resolve()
        behavior_change_scale_path = Path(SCORER_LIKERT_PATH, "behavior_change.yaml").resolve()
        behavior_change_scale = yaml.safe_load(behavior_change_scale_path.read_text(encoding="utf-8"))

        scoring_instructions_template = SeedPrompt.from_yaml_file(behavior_change_prompt_path)

        self._system_prompt = scoring_instructions_template.render_template_value(
            step_description=behavior_change_scale
        )

    async def _score_piece_async(self, message_piece: MessagePiece, *, objective: Optional[str] = None) -> list[Score]:
        """
        Scores the entire conversation based on detected behavior change.

        Args:
            message_piece (MessagePiece): A piece of the conversation to be scored.
                The conversation ID is used to retrieve the full conversation from memory.
            objective (Optional[str]): The objective to evaluate against. Defaults to None.
                Currently not used for this scorer.

        Returns:
            list[Score]: A list containing a single Score object representing the detected
                amount of behavior change throughout the conversation.

        Raises:
            ValueError: If the conversation ID is not found in memory.
        """
        # Retrieve conversation from provided MessagePiece
        conversation = self._memory.get_conversation(conversation_id=message_piece.conversation_id)

        if not conversation:
            raise ValueError(f"Conversation with ID {message_piece.conversation_id} not found in memory.")

        # Loop through each request to create a single string containing the conversation text
        conversation_as_text = ""
        for request in conversation:
            # Check exclude_instruction_prompts flag
            if self.exclude_instruction_prompts:
                # Skip instruction prompts
                if re.search(r"#\s*Instructions", request.message_pieces[0].original_value, re.IGNORECASE):
                    continue
            # Append the request text to the conversation string
            # Only append user and assistant roles to the conversation string
            if request.message_pieces[0].role in ["user", "assistant"]:
                conversation_as_text += (
                    f"{request.message_pieces[0].role}: {request.message_pieces[0].original_value}\n"
                )

        # Send full conversation to LLM for scoring
        unvalidated_score: UnvalidatedScore = await self._score_value_with_llm(
            prompt_target=self._prompt_target,
            system_prompt=self._system_prompt,
            message_value=conversation_as_text,
            message_data_type="text",
            scored_prompt_id=conversation[0].message_pieces[0].original_prompt_id,
            category="illegal",
            attack_identifier=conversation[0].message_pieces[0].attack_identifier,
        )

        score = unvalidated_score.to_score(score_value=unvalidated_score.raw_score_value, score_type="float_scale")

        # Output score results
        logger.info(
            "LookBackScorer:", score.score_value, score.score_value_description, "Rationale: ", score.score_rationale
        )
        return [score]

    def _get_scorer_specific_params(self):
        scorer_specific_params = super()._get_scorer_specific_params()
        return {
            **(scorer_specific_params or {}),
            "exclude_instruction_prompts": self.exclude_instruction_prompts,
        }
