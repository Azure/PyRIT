# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
from typing import Optional, cast

from typing_extensions import LiteralString, deprecated

from pyrit.common import deprecation_message
from pyrit.executor.attack import (
    AttackConverterConfig,
    AttackScoringConfig,
    ManyShotJailbreakAttack,
)
from pyrit.orchestrator import PromptSendingOrchestrator
from pyrit.orchestrator.models.orchestrator_result import OrchestratorResult
from pyrit.prompt_normalizer import PromptConverterConfiguration
from pyrit.prompt_target import PromptTarget
from pyrit.score import Scorer

logger = logging.getLogger(__name__)


@deprecated(
    cast(
        LiteralString,
        deprecation_message(
            old_item="ManyShotJailbreakOrchestrator",
            new_item=ManyShotJailbreakAttack,
            removed_in="v0.12.0",
        ),
    ),
)
class ManyShotJailbreakOrchestrator(PromptSendingOrchestrator):
    """
    .. warning::
        `ManyShotJailbreakOrchestrator` is deprecated and will be removed in **v0.12.0**;
        use `pyrit.executor.attack.ManyShotJailbreakAttack` instead.

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
        example_count: int = 100,
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
            example_count (int): The number of examples to include from the examples dataset.
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

        self._attack = ManyShotJailbreakAttack(
            objective_target=objective_target,
            attack_converter_config=AttackConverterConfig(
                request_converters=self._request_converter_configurations,
                response_converters=self._response_converter_configurations,
            ),
            attack_scoring_config=AttackScoringConfig(
                objective_scorer=self._objective_scorer,
                auxiliary_scorers=self._auxiliary_scorers,
            ),
            prompt_normalizer=self._prompt_normalizer,
            max_attempts_on_failure=retries_on_objective_failure,
            example_count=example_count,
            many_shot_examples=many_shot_examples,
        )

    async def run_attack_async(  # type: ignore[override]
        self,
        *,
        objective: str,
        memory_labels: Optional[dict[str, str]] = None,
    ) -> OrchestratorResult:
        return await super().run_attack_async(
            objective=objective,
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
