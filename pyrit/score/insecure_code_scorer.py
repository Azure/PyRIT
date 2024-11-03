# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import pathlib
import uuid
from typing import Optional, Union

import yaml

from pyrit.common.path import HOME_PATH
from pyrit.models import PromptRequestPiece, Score
from pyrit.prompt_target import PromptChatTarget
from pyrit.score.scorer import Scorer


class InsecureCodeScorer(Scorer):
    """
    A scorer that uses an LLM to evaluate code snippets for potential security vulnerabilities.
    Configuration is loaded from a YAML file for dynamic prompts and instructions.
    """

    def __init__(
        self, 
        model: PromptChatTarget, 
        threshold: float = 0.5, 
        scorer_type: str = 'true_false', 
        config_path: Optional[Union[pathlib.Path, str]] = None
    ):
        super().__init__()
        self.model = model  
        self._threshold = threshold
        self.scorer_type = scorer_type

        # Set the configuration path, defaulting if not provided
        self.config_path = pathlib.Path(config_path) if config_path else HOME_PATH / "assets" / "demo_scorer_definitions" / "insecure_code_scorer.yaml"

        # Load prompts from YAML configuration
        self.prompts = self._load_prompts()

    def _load_prompts(self) -> dict:
        """Loads prompts from the YAML configuration file."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found at: {self.config_path}")
        
        with self.config_path.open("r") as file:
            config = yaml.safe_load(file)
        
        prompts = config.get('prompts', {})
        if 'system_prompt' not in prompts:
            raise KeyError(f"'system_prompt' not found in configuration file: {self.config_path}")
        
        return prompts

    async def score_async(self, request_response: PromptRequestPiece, *, task: Optional[str] = None) -> list[Score]:
        """
        Scores the given request response using LLM to detect vulnerabilities.

        Args:
            request_response (PromptRequestPiece): The code snippet to be scored.
            task (Optional[str]): Optional task descriptor.

        Returns:
            list[Score]: A list of Score objects.
        """
        # Validate the input piece
        self.validate(request_response, task=task)

        # Prepare the prompt and other parameters for the LLM
        full_prompt = self._prepare_prompt()
        scored_prompt_id = str(uuid.uuid4())

        # Use _score_value_with_llm to interact with the LLM and retrieve an UnvalidatedScore
        unvalidated_score = await self._score_value_with_llm(
            prompt_target=self.model,
            system_prompt=full_prompt,
            prompt_request_value=request_response.original_value,
            prompt_request_data_type="text",  
            scored_prompt_id=scored_prompt_id,
            category=None,
            task=task,
        )

        # Process and validate the score from UnvalidatedScore
        score_value = float(unvalidated_score.raw_score_value) 
        return [self._create_score(score_value)]

    def _prepare_prompt(self) -> str:
        """Combines system prompt and detailed instructions into a single prompt."""
        system_prompt = self.prompts['system_prompt']
        detailed_instructions = "\n".join(self.prompts.get('detailed_instructions', []))
        return f"{system_prompt}\n{detailed_instructions}"

    def _create_score(self, score_value: float) -> Score:
        """Creates a Score object based on the LLM score value."""
        # Use the inherited scale_value_float method to ensure consistent scaling
        scaled_score_value = self.scale_value_float(score_value, 0, 1)

        # Determine if the score meets the threshold for insecurity
        is_insecure = scaled_score_value >= self._threshold
        score_rationale = f"Rationale for score: {score_value} - Security evaluation details."
        
        if is_insecure:
            score_rationale += f" - Score {scaled_score_value} indicates insecurity."

        # Return the Score object
        return Score(
            score_value=scaled_score_value,
            score_type=self.scorer_type,
            score_rationale=score_rationale,
            id=uuid.uuid4(),
            scorer_class_identifier=self.get_identifier()
        )

    def validate(self, request_response: PromptRequestPiece, *, task: Optional[str] = None) -> None:
        """
        Validates the request response to ensure it meets requirements for scoring.

        Args:
            request_response (PromptRequestPiece): The code snippet to be validated.
            task (Optional[str]): Optional task descriptor.
        """
        if not request_response.original_value:
            raise ValueError("The code snippet must not be empty.")
