# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import enum
import logging
import pathlib
from typing import Optional, cast

from typing_extensions import LiteralString, deprecated

from pyrit.common import deprecation_message
from pyrit.common.path import DATASETS_PATH
from pyrit.executor.attack import (
    AttackAdversarialConfig,
    AttackConverterConfig,
    AttackScoringConfig,
    ContextComplianceAttack,
)
from pyrit.orchestrator import OrchestratorResult, PromptSendingOrchestrator
from pyrit.prompt_normalizer.prompt_converter_configuration import (
    PromptConverterConfiguration,
)
from pyrit.prompt_target import PromptChatTarget
from pyrit.score import Scorer

logger = logging.getLogger(__name__)


class ContextDescriptionPaths(enum.Enum):
    GENERAL = pathlib.Path(DATASETS_PATH) / "executors" / "context_compliance" / "context_description.yaml"


@deprecated(
    cast(
        LiteralString,
        deprecation_message(
            old_item="ContextComplianceOrchestrator",
            new_item=ContextComplianceAttack,
            removed_in="v0.12.0",
        ),
    ),
)
class ContextComplianceOrchestrator(PromptSendingOrchestrator):
    """
    .. warning::
        `ContextComplianceOrchestrator` is deprecated and will be removed in **v0.12.0**;
        use `pyrit.executor.attack.ContextComplianceAttack` instead.

    This orchestrator implements a context compliance attack that attempts to bypass safety measures by
    rephrasing the objective into a more benign context. It uses an adversarial chat target to:
    1. Rephrase the objective as a more benign question
    2. Generate a response to the benign question
    3. Rephrase the original objective as a follow-up question
    This creates a context that makes it harder for the target to detect the true intent.
    """

    def __init__(
        self,
        objective_target: PromptChatTarget,
        adversarial_chat: PromptChatTarget,
        affirmative_response: str = "yes.",
        context_description_instructions_path: Optional[pathlib.Path] = None,
        request_converter_configurations: Optional[list[PromptConverterConfiguration]] = None,
        response_converter_configurations: Optional[list[PromptConverterConfiguration]] = None,
        objective_scorer: Optional[Scorer] = None,
        auxiliary_scorers: Optional[list[Scorer]] = None,
        batch_size: int = 10,
        retries_on_objective_failure: int = 0,
        verbose: bool = False,
    ) -> None:
        """
        Args:
            objective_target (PromptChatTarget): The target for sending prompts.
            adversarial_chat (PromptChatTarget): The target used to rephrase objectives into benign contexts.
            affirmative_response (str, Optional): The affirmative response to be used in the conversation history.
            context_description_instructions_path (pathlib.Path, Optional): Path to the context description
                instructions YAML file.
            request_converter_configurations (list[PromptConverterConfiguration], Optional): List of prompt
                converters.
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

        # Use default path if not provided
        if context_description_instructions_path is None:
            context_description_instructions_path = ContextDescriptionPaths.GENERAL.value

        self._attack = ContextComplianceAttack(
            objective_target=objective_target,
            attack_adversarial_config=AttackAdversarialConfig(target=adversarial_chat),
            attack_converter_config=AttackConverterConfig(
                request_converters=self._request_converter_configurations,
                response_converters=self._response_converter_configurations,
            ),
            attack_scoring_config=AttackScoringConfig(
                objective_scorer=self._objective_scorer,
                auxiliary_scorers=self._auxiliary_scorers,
            ),
            prompt_normalizer=self._prompt_normalizer,
            context_description_instructions_path=context_description_instructions_path,
            affirmative_response=affirmative_response,
            max_attempts_on_failure=self._retries_on_objective_failure,
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
