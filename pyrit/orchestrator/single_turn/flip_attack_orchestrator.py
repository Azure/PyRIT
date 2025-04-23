# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
import pathlib
from typing import Optional, Union

from pyrit.orchestrator.models.orchestrator_result import OrchestratorResult
from pyrit.common.path import DATASETS_PATH
from pyrit.models import PromptRequestResponse, SeedPrompt, SeedPromptGroup
from pyrit.models.prompt_request_piece import PromptRequestPiece
from pyrit.orchestrator import PromptSendingOrchestrator
from pyrit.prompt_converter import FlipConverter, PromptConverter
from pyrit.prompt_normalizer import NormalizerRequest, PromptConverterConfiguration
from pyrit.prompt_target import PromptChatTarget
from pyrit.score import Scorer

logger = logging.getLogger(__name__)


class FlipAttackOrchestrator(PromptSendingOrchestrator):
    """
    This orchestrator implements the Flip Attack method found here:
    https://arxiv.org/html/2410.02832v1.

    Essentially, adds a system prompt to the beginning of the conversation to flip each word in the prompt.
    """

    def __init__(
        self,
        objective_target: PromptChatTarget,
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
            prompt_converters (list[PromptConverter], Optional): List of prompt converters. These are stacked in
                order on top of the flip converter.
            scorers (list[Scorer], Optional): List of scorers to use for each prompt request response, to be
                scored immediately after receiving response. Default is None.
            batch_size (int, Optional): The (max) batch size for sending prompts. Defaults to 10.
                Note: If providing max requests per minute on the objective_target, this should be set to 1 to
                ensure proper rate limit management.\
            verbose (bool, Optional): Whether to log debug information. Defaults to False.
        """

        flip_converter = PromptConverterConfiguration.create_configurations_from_converters([FlipConverter()])


        super().__init__(
            objective_target=objective_target,
            request_converter_configurations= flip_converter + (request_converter_configurations or []),
            response_converter_configurations=response_converter_configurations,
            objective_scorer=objective_scorer,
            auxiliary_scorers=auxiliary_scorers,
            should_convert_prepended_conversation=False,
            batch_size=batch_size,
            retries_on_objective_failure=retries_on_objective_failure,
            verbose=verbose,
        )

        # This system prompt is sent to the target
        system_prompt_path = pathlib.Path(DATASETS_PATH) / "orchestrators" / "flip_attack.yaml"
        system_prompt = SeedPrompt.from_yaml_file(system_prompt_path).value

        self._system_prompt = PromptRequestResponse.get_system_prompt_request(system_prompt)



    async def run_attack_async(
        self,
        *,
        objective: str,
        memory_labels: Optional[dict[str, str]] = None,
    ) -> OrchestratorResult:
        
        initial_prompt = "TASK is '⟪" + objective.replace("'", "") + "⟫'"
        seed_prompt = SeedPromptGroup(prompts=[SeedPrompt(value=initial_prompt, data_type="text")])
        
        return await super().run_attack_async(
            objective=objective,
            seed_prompt=seed_prompt,
            prepended_conversation=self._system_prompt,
            memory_labels=memory_labels,
        )


    async def run_attacks_async(
        self,
        *,
        objectives: list[str],
        memory_labels: Optional[dict[str, str]] = None,
    ) -> list[OrchestratorResult]:
        return await super()._run_attacks_with_only_objectives_async(
            objectives=objectives,
            memory_labels=memory_labels,
        )
