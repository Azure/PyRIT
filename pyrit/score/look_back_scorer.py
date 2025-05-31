# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import re
from pathlib import Path
from typing import Optional

import yaml

from pyrit.common.path import DATASETS_PATH
from pyrit.models import PromptRequestPiece, SeedPrompt
from pyrit.models.score import Score, UnvalidatedScore
from pyrit.prompt_target import PromptChatTarget
from pyrit.score.scorer import Scorer


class LookBackScorer(Scorer):
    """
    Create a score from analyzing the entire conversation and adds them to the database.

    Parameters:
        chat_target (PromptChatTarget): The chat target to use for scoring.
        exclude_instruction_prompts (bool): If True, excludes instruction prompts from the conversation.
            Must be passed as a keyword argument.
    """

    def __init__(self, chat_target: PromptChatTarget, *, exclude_instruction_prompts: bool) -> None:
        self._prompt_target = chat_target
        self.scorer_type = "float_scale"
        self.exclude_instruction_prompts = exclude_instruction_prompts

        behavior_change_prompt_path = Path(
            DATASETS_PATH, "score", "scales", "behavior_change_system_prompt.yaml"
        ).resolve()
        behavior_change_scale_path = Path(DATASETS_PATH, "score", "likert_scales", "behavior_change.yaml")
        behavior_change_scale = yaml.safe_load(behavior_change_scale_path.read_text(encoding="utf-8"))

        scoring_instructions_template = SeedPrompt.from_yaml_file(behavior_change_prompt_path)

        self._system_prompt = scoring_instructions_template.render_template_value(
            step_description=behavior_change_scale
        )

    async def score_async(self, request_piece: PromptRequestPiece, *, task: Optional[str] = None) -> list[Score]:
        """
        Scores the entire conversation based on detected behavior change.

        Args:
            request_piece (PromptRequestPiece): A piece of the conversation to be scored.
                The converation ID is used to retrieve the full conversation from memory.
            task (str): The task based on which the text should be scored (the original attacker model's objective).
                Currently not supported for this scorer.

        Returns:
            The score is the detected amount of behavior change throughout the conversation.
        """

        # Retrieve conversation from provided PromptRequestPiece
        conversation = self._memory.get_conversation(conversation_id=request_piece.conversation_id)

        if not conversation:
            raise ValueError(f"Conversation with ID {request_piece.conversation_id} not found in memory.")

        # Loop through each request to create a single string containing the conversation text
        conversation_as_text = ""
        for request in conversation:
            # Check exclude_instruction_prompts flag
            if self.exclude_instruction_prompts:
                # Skip instruction prompts
                if re.search(r"#\s*Instructions", request.request_pieces[0].original_value, re.IGNORECASE):
                    continue
            # Append the request text to the conversation string
            # Only append user and assistant roles to the conversation string
            if request.request_pieces[0].role in ["user", "assistant"]:
                conversation_as_text += (
                    f"{request.request_pieces[0].role}: {request.request_pieces[0].original_value}\n"
                )

        # Send full conversation to LLM for scoring
        unvalidated_score: UnvalidatedScore = await self._score_value_with_llm(
            prompt_target=self._prompt_target,
            system_prompt=self._system_prompt,
            prompt_request_value=conversation_as_text,
            prompt_request_data_type="text",
            scored_prompt_id=conversation[0].request_pieces[0].original_prompt_id,
            category="illegal",
            orchestrator_identifier=conversation[0].request_pieces[0].orchestrator_identifier,
        )

        score = unvalidated_score.to_score(score_value=unvalidated_score.raw_score_value)
        self._memory.add_scores_to_memory(scores=[score])

        # Output score results
        print("LookBackScorer:", score.score_value, score.score_value_description, "Rationale: ", score.score_rationale)
        return [score]

    def validate(self, request_response: PromptRequestPiece, *, task: Optional[str] = None):
        pass
