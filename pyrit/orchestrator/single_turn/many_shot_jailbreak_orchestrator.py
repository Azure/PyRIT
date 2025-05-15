# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
from pathlib import Path
from typing import Optional

from pyrit.common.path import DATASETS_PATH
from pyrit.datasets import fetch_many_shot_jailbreaking_dataset
from pyrit.models import SeedPrompt, SeedPromptGroup
from pyrit.orchestrator import PromptSendingOrchestrator
from pyrit.orchestrator.models.orchestrator_result import OrchestratorResult
from pyrit.prompt_normalizer import PromptConverterConfiguration
from pyrit.prompt_target import PromptTarget
from pyrit.score import Scorer

logger = logging.getLogger(__name__)


class ManyShotJailbreakOrchestrator(PromptSendingOrchestrator):
    """
    This orchestrator implements the Many Shot Jailbreak method as discussed in research found here:
    https://www.anthropic.com/research/many-shot-jailbreaking

    Prepends the seed prompt with a faux dialogue between a human and an AI, using examples from a dataset
    to demonstrate successful jailbreaking attempts. This method leverages the model's ability to learn from
    examples to bypass safety measures.
    """

    def __init__(
        self,
        objective_target: PromptTarget,
        request_converter_configurations: Optional[list[PromptConverterConfiguration]] = None,
        response_converter_configurations: Optional[list[PromptConverterConfiguration]] = None,
        objective_scorer: Optional[Scorer] = None,
        auxiliary_scorers: Optional[list[Scorer]] = None,
        batch_size: int = 10,
        retries_on_objective_failure: int = 0,
        verbose: bool = False,
        example_count: Optional[int] = 100,
        many_shot_examples: Optional[list[dict[str, str]]] = None,
    ) -> None:
        """
        Args:
            objective_target (PromptTarget): The target for sending prompts.
            request_converter_configurations (list[PromptConverterConfiguration], Optional): List of prompt converters.
            response_converter_configurations (list[PromptConverterConfiguration], Optional): List of response
                converters.
            objective_scorer (Scorer, Optional): Scorer to use for evaluating if the objective was achieved.
            auxiliary_scorers (list[Scorer], Optional): List of additional scorers to use for each prompt request
                response.
            batch_size (int, Optional): The (max) batch size for sending prompts. Defaults to 10.
                Note: If providing max requests per minute on the prompt_target, this should be set to 1 to
                ensure proper rate limit management.
            retries_on_objective_failure (int, Optional): Number of retries to attempt if objective fails. Defaults to
                0.
            verbose (bool, Optional): Whether to log debug information. Defaults to False.
            example_count (int, Optional): The number of examples to include from the examples dataset.
                Defaults to the first 100.
            many_shot_examples (list[dict[str, str]], Optional): The many shot jailbreaking examples to use.
                If not provided, takes the first `example_count` examples from Many Shot Jailbreaking dataset.
        """

        super().__init__(
            objective_target=objective_target,
            request_converter_configurations=request_converter_configurations,
            response_converter_configurations=response_converter_configurations,
            objective_scorer=objective_scorer,
            auxiliary_scorers=auxiliary_scorers,
            should_convert_prepended_conversation=True,
            batch_size=batch_size,
            retries_on_objective_failure=retries_on_objective_failure,
            verbose=verbose,
        )

        # Template for the faux dialogue to be prepended
        template_path = Path(DATASETS_PATH) / "prompt_templates" / "jailbreak" / "many_shot_template.yaml"
        self._template = SeedPrompt.from_yaml_file(template_path)
        # Fetch the Many Shot Jailbreaking example dataset
        self._examples = (
            many_shot_examples[:example_count]
            if (many_shot_examples is not None)
            else fetch_many_shot_jailbreaking_dataset()[:example_count]
        )
        if not self._examples:
            raise ValueError("Many shot examples must be provided.")

    async def run_attack_async(  # type: ignore[override]
        self,
        *,
        objective: str,
        memory_labels: Optional[dict[str, str]] = None,
    ) -> OrchestratorResult:

        many_shot_prompt = self._template.render_template_value(prompt=objective, examples=self._examples)

        seed_prompt = SeedPromptGroup(prompts=[SeedPrompt(value=many_shot_prompt, data_type="text")])

        return await super().run_attack_async(
            objective=objective,
            seed_prompt=seed_prompt,
            memory_labels=memory_labels,
        )

    async def run_attacks_async(  # type: ignore[override]
        self,
        *,
        objectives: list[str],
        memory_labels: Optional[dict[str, str]] = None,
    ) -> list[OrchestratorResult]:
        return await super()._run_attacks_with_only_objectives_async(
            objectives=objectives,
            memory_labels=memory_labels,
        )
