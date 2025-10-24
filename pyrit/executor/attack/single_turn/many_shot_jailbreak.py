# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
import pathlib
from typing import Optional

from pyrit.common.apply_defaults import apply_defaults
from pyrit.common.path import DATASETS_PATH
from pyrit.datasets import fetch_many_shot_jailbreaking_dataset
from pyrit.executor.attack.core import AttackConverterConfig, AttackScoringConfig
from pyrit.executor.attack.single_turn import SingleTurnAttackContext
from pyrit.executor.attack.single_turn.prompt_sending import PromptSendingAttack
from pyrit.models import AttackResult, SeedPrompt, SeedPromptGroup
from pyrit.prompt_normalizer import PromptNormalizer
from pyrit.prompt_target import PromptTarget

logger = logging.getLogger(__name__)


class ManyShotJailbreakAttack(PromptSendingAttack):
    """
    This attack implements implements the Many Shot Jailbreak method as discussed in research found here:
    https://www.anthropic.com/research/many-shot-jailbreaking

    Prepends the seed prompt with a faux dialogue between a human and an AI, using examples from a dataset
    to demonstrate successful jailbreaking attempts. This method leverages the model's ability to learn from
    examples to bypass safety measures.
    """

    @apply_defaults
    def __init__(
        self,
        objective_target: PromptTarget,
        attack_converter_config: Optional[AttackConverterConfig] = None,
        attack_scoring_config: Optional[AttackScoringConfig] = None,
        prompt_normalizer: Optional[PromptNormalizer] = None,
        max_attempts_on_failure: int = 0,
        example_count: int = 100,
        many_shot_examples: Optional[list[dict[str, str]]] = None,
    ) -> None:
        """
        Args:
            objective_target (PromptTarget): The target system to attack.
            attack_converter_config (AttackConverterConfig, Optional): Configuration for the prompt converters.
            attack_scoring_config (AttackScoringConfig, Optional): Configuration for scoring components.
            prompt_normalizer (PromptNormalizer, Optional): Normalizer for handling prompts.
            max_attempts_on_failure (int, Optional): Maximum number of attempts to retry on failure. Defaults to 0.
            example_count (int): The number of examples to include from many_shot_examples or the Many
                Shot Jailbreaking dataset. Defaults to the first 100.
            many_shot_examples (list[dict[str, str]], Optional): The many shot jailbreaking examples to use.
                If not provided, takes the first `example_count` examples from Many Shot Jailbreaking dataset.
        """
        super().__init__(
            objective_target=objective_target,
            attack_converter_config=attack_converter_config,
            attack_scoring_config=attack_scoring_config,
            prompt_normalizer=prompt_normalizer,
            max_attempts_on_failure=max_attempts_on_failure,
        )

        # Template for the faux dialogue to be prepended
        template_path = pathlib.Path(DATASETS_PATH) / "jailbreak" / "multi_parameter" / "many_shot_template.yaml"
        self._template = SeedPrompt.from_yaml_file(template_path)
        # Fetch the Many Shot Jailbreaking example dataset
        self._examples = (
            many_shot_examples[:example_count]
            if (many_shot_examples is not None)
            else fetch_many_shot_jailbreaking_dataset()[:example_count]
        )
        if not self._examples:
            raise ValueError("Many shot examples must be provided.")

    def _validate_context(self, *, context: SingleTurnAttackContext) -> None:
        """
        Validate the context before executing the attack.
        Args:
            context (SingleTurnAttackContext): The attack context containing parameters and objective.
        Raises:
            ValueError: If the context is invalid.
        """
        if context.prepended_conversation:
            raise ValueError("ManyShotJailbreakAttack does not support prepended conversations.")
        super()._validate_context(context=context)

    async def _perform_async(self, *, context: SingleTurnAttackContext) -> AttackResult:
        """
        Perform the ManyShotJailbreakAttack.
        Args:
            context (SingleTurnAttackContext): The attack context containing attack parameters.
        Returns:
            AttackResult: The result of the attack.
        """
        many_shot_prompt = self._template.render_template_value(prompt=context.objective, examples=self._examples)
        seed_prompt_group = SeedPromptGroup(prompts=[SeedPrompt(value=many_shot_prompt, data_type="text")])

        context.seed_prompt_group = seed_prompt_group

        return await super()._perform_async(context=context)
