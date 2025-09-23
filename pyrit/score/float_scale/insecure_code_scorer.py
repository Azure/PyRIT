# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from pathlib import Path
from typing import Optional, Union

from pyrit.common.path import SCORER_CONFIG_PATH
from pyrit.exceptions.exception_classes import InvalidJsonException
from pyrit.models import PromptRequestPiece, Score, SeedPrompt
from pyrit.prompt_target import PromptChatTarget
from pyrit.score.float_scale.float_scale_scorer import FloatScaleScorer
from pyrit.score.scorer_prompt_validator import ScorerPromptValidator


class InsecureCodeScorer(FloatScaleScorer):
    """
    A scorer that uses an LLM to evaluate code snippets for potential security vulnerabilities.
    Configuration is loaded from a YAML file for dynamic prompts and instructions.
    """

    _default_validator: ScorerPromptValidator = ScorerPromptValidator(supported_data_types=["text"])

    def __init__(
        self,
        *,
        chat_target: PromptChatTarget,
        system_prompt_path: Optional[Union[str, Path]] = None,
        validator: Optional[ScorerPromptValidator] = None,
    ):
        super().__init__(validator=validator or self._default_validator)

        if not system_prompt_path:
            system_prompt_path = SCORER_CONFIG_PATH / "insecure_code" / "system_prompt.yaml"

        self._system_prompt_path: Path = self._verify_and_resolve_path(system_prompt_path)
        self._prompt_target = chat_target

        # Load the system prompt template as a SeedPrompt object
        scoring_instructions_template = SeedPrompt.from_yaml_file(self._system_prompt_path)

        # Define the harm category
        self._harm_category = "security"

        # Render the system prompt with the harm category
        self._system_prompt = scoring_instructions_template.render_template_value(harm_categories=self._harm_category)

    async def _score_piece_async(
        self, request_piece: PromptRequestPiece, *, objective: Optional[str] = None
    ) -> list[Score]:
        """
        Scores the given request response using LLM to detect vulnerabilities.

        Args:
            request_response (PromptRequestPiece): The code snippet to be scored.
            task (Optional[str]): Optional task descriptor.

        Returns:
            list[Score]: A list of Score objects.
        """
        # Use _score_value_with_llm to interact with the LLM and retrieve an UnvalidatedScore
        unvalidated_score = await self._score_value_with_llm(
            prompt_target=self._prompt_target,
            system_prompt=self._system_prompt,
            prompt_request_value=request_piece.original_value,
            prompt_request_data_type=request_piece.converted_value_data_type,
            scored_prompt_id=request_piece.id,
            category=self._harm_category,
            objective=objective,
            orchestrator_identifier=request_piece.orchestrator_identifier,
        )

        # Modify the UnvalidatedScore parsing to check for 'score_value'
        try:
            # Attempt to use score_value if available
            raw_score_value = float(unvalidated_score.raw_score_value)
        except KeyError:
            raise InvalidJsonException(message="Expected 'score_value' key missing in the JSON response")

        # Convert UnvalidatedScore to Score, applying scaling and metadata
        score = unvalidated_score.to_score(
            score_value=str(self.scale_value_float(raw_score_value, 0, 1)),
            score_type="float_scale",
        )

        return [score]
