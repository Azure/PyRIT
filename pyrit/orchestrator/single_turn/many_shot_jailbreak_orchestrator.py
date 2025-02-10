# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
import random
from typing import Optional
from pathlib import Path

from pyrit.common.path import DATASETS_PATH 
from pyrit.datasets import fetch_many_shot_jailbreaking_dataset
from pyrit.models import SeedPrompt, PromptRequestResponse
from pyrit.orchestrator import PromptSendingOrchestrator
from pyrit.prompt_target import PromptChatTarget
from pyrit.score import Scorer

logger = logging.getLogger(__name__)

class ManyShotJailbreakOrchestrator(PromptSendingOrchestrator):
    """
    This orchestrator implements the Many Shot Jailbreak methos as discussed in research found here: https://www.anthropic.com/research/many-shot-jailbreaking
    
    Prepends the seed prompt with a faux dialogue between a human and an AI
    """
    def __init__(
            self,
            objective_target: PromptChatTarget,
            scorers: Optional[list[Scorer]] = None,
            verbose: bool = False,
            num_examples: int = 3,
            isTest: Optional[bool] = False
    ) -> None:
        """
        Args:
            objective_target: (PromptChatTarget): The target for sending prompts.
            scorers (list[Scorer], Optional): List of scorers to use for each prompt request response, to be
                scored immediately after receiving response. Default is None.
            verbose (bool, Optional): Whether to log debug information. Defaults to False.
            num_examples (int, Optional): The number of examples to fetch from the dataset. Defaults to 3.
            isTest (bool, Optional): Controls whether or not the examples are gotten at random or statically. Setting to True can be useful for testing. Defaults to False.
        """
        
        super().__init__(
            objective_target=objective_target,
            scorers=scorers,
            verbose=verbose,
        )

        # Template for the faux dialogue to be prepended
        template_path = Path(DATASETS_PATH) / "prompt_templates" / "jailbreak" / "many_shot_template.yaml"
        self.template = SeedPrompt.from_yaml_file(template_path)
        self.num_examples = num_examples
        self.isTest = isTest

    async def construct_many_shot_dialogue(
            self,
            malicious_prompt: str
    ) -> str:
        """
        Constructs the many shot dialogue to be prepended to the prompt

        Args:
            malicious_prompt (str): The malicious prompt to be prepended

        Returns:
            str: The constructed many shot dialogue
        """
        # Fetch the Many Shot Jailbreaking dataset
        examples = fetch_many_shot_jailbreaking_dataset()

        # Choose a random subset of examples of size num_examples, check that the number of examples is not greater than the number of examples in the dataset
        if (self.num_examples > len(examples)):
            logger.info(f"Specified example count is greater than the number of examples in the dataset. Using all {len(examples)} examples.")
            self.num_examples = len(examples)

        # Choose num_examples either static or random examples from the dataset
        examples = (examples[:self.num_examples] if (self.isTest) else random.sample(examples, self.num_examples))

        # Construct the many shot dialogue
        many_shot_dialogue = self.template.render_template_value(prompt = malicious_prompt, examples = examples)

        return many_shot_dialogue


    async def send_prompts_async(
            self,
            *,
            prompt_list: list[str],
            memory_labels: Optional[dict[str, str]] = None,
            metadata: Optional[dict[str, str]] = None,
    ) -> list[PromptRequestResponse]:
        """
        Sends the prompts to the target using the Many Shot Jailbreak

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
        for i in range(0, len(prompt_list)):
            prompt_list[i] = await self.construct_many_shot_dialogue(prompt_list[i])

        return await super().send_prompts_async(
            prompt_list=prompt_list, prompt_type = "text", memory_labels=memory_labels, metadata=metadata
        )