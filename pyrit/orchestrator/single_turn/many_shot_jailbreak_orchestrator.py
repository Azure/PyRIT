# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
from pathlib import Path
from typing import Optional, Union

from pyrit.common.path import DATASETS_PATH
from pyrit.datasets import fetch_many_shot_jailbreaking_dataset
from pyrit.models import PromptRequestResponse, SeedPrompt
from pyrit.orchestrator import PromptSendingOrchestrator
from pyrit.prompt_target import PromptChatTarget
from pyrit.score import Scorer

logger = logging.getLogger(__name__)


class ManyShotJailbreakOrchestrator(PromptSendingOrchestrator):
    """
    This orchestrator implements the Many Shot Jailbreak method as discussed in research found here:
    https://www.anthropic.com/research/many-shot-jailbreaking

    Prepends the seed prompt with a faux dialogue between a human and an AI.
    """

    def __init__(
        self,
        objective_target: PromptChatTarget,
        scorers: Optional[list[Scorer]] = None,
        verbose: bool = False,
        example_count: Optional[int] = 100,
        many_shot_examples: Optional[list[dict[str, str]]] = None,
    ) -> None:
        """
        Args:
            objective_target: (PromptChatTarget): The target for sending prompts.
            scorers (list[Scorer], Optional): List of scorers to use for each prompt request response, to be
                scored immediately after receiving response. Default is None.
            verbose (bool, Optional): Whether to log debug information. Defaults to False.
            example_count (int, Optional): The number of examples to include from the Many Shot Jailbreaking
                dataset. Defaults to the first 100.
            many_shot_examples (list[dict[str, str]], Optional): The many shot jailbreaking examples to use.
                If not provided, uses all examples from the Many Shot Jailbreaking dataset.
        """

        super().__init__(
            objective_target=objective_target,
            scorers=scorers,
            verbose=verbose,
        )

        # Template for the faux dialogue to be prepended
        template_path = Path(DATASETS_PATH) / "prompt_templates" / "jailbreak" / "many_shot_template.yaml"
        self._template = SeedPrompt.from_yaml_file(template_path)
        # Fetch the Many Shot Jailbreaking example dataset
        self._examples = (
            many_shot_examples
            if (many_shot_examples is not None)
            else fetch_many_shot_jailbreaking_dataset()[:example_count]
        )
        if not self._examples:
            raise ValueError("Many shot examples must be provided.")

    async def send_prompts_async(  # type: ignore[override]
        self,
        *,
        prompt_list: list[str],
        memory_labels: Optional[dict[str, str]] = None,
        metadata: Optional[dict[str, Union[str, int]]] = None,
    ) -> list[PromptRequestResponse]:
        """
        Sends the prompts to the target using the Many Shot Jailbreak.

        Args:
            prompt_list (list[str]): The list of prompts to be sent.
            memory_labels (dict[str, str], Optional): A free-form dictionary of additional labels to apply to the
                prompts. Any labels passed in will be combined with self._global_memory_labels with the passed
                in labels taking precedence in the case of collisions. Defaults to None.
            metadata (Optional(dict[str, str]): Any additional information to be added to the memory entry corresponding
                to the prompts sent.

        Returns:
            list[PromptRequestResponse]: The responses from sending the prompts.
        """
        if not prompt_list:
            raise ValueError("Prompt list must not be empty.")

        many_shot_prompt_list = [
            self._template.render_template_value(prompt=prompt, examples=self._examples) for prompt in prompt_list
        ]

        return await super().send_prompts_async(
            prompt_list=many_shot_prompt_list, prompt_type="text", memory_labels=memory_labels, metadata=metadata
        )
