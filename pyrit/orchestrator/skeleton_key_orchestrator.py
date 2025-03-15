# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
from pathlib import Path
from typing import Optional
from uuid import uuid4

from colorama import Fore, Style

from pyrit.common.path import DATASETS_PATH
from pyrit.models import (
    PromptRequestResponse,
    SeedPrompt,
    SeedPromptDataset,
    SeedPromptGroup,
)
from pyrit.orchestrator import Orchestrator
from pyrit.prompt_converter import PromptConverter
from pyrit.prompt_normalizer import PromptNormalizer
from pyrit.prompt_normalizer.prompt_converter_configuration import (
    PromptConverterConfiguration,
)
from pyrit.prompt_target import PromptTarget
from pyrit.prompt_target.batch_helper import batch_task_async

logger = logging.getLogger(__name__)


class SkeletonKeyOrchestrator(Orchestrator):
    """
    Creates an orchestrator that executes a skeleton key jailbreak.

    The orchestrator sends an inital skeleton key prompt to the target, and then follows
    up with a separate attack prompt.
    If successful, the first prompt makes the target comply even with malicious follow-up prompts.
    In our experiments, using two separate prompts was more effective than using a single combined prompt.

    Learn more about attack at the link below:
    https://www.microsoft.com/en-us/security/blog/2024/06/26/mitigating-skeleton-key-a-new-type-of-generative-ai-jailbreak-technique/
    """

    def __init__(
        self,
        *,
        skeleton_key_prompt: Optional[str] = None,
        prompt_target: PromptTarget,
        prompt_converters: Optional[list[PromptConverter]] = None,
        batch_size: int = 10,
        verbose: bool = False,
    ) -> None:
        """
        Args:
            skeleton_key_prompt (str, Optional): The skeleton key sent to the target, Default: skeleton_key.prompt
            prompt_target (PromptTarget): The target for sending prompts.
            prompt_converters (list[PromptConverter], Optional): List of prompt converters. These are stacked in
                the order they are provided. E.g. the output of converter1 is the input of converter2.
            batch_size (int, Optional): The (max) batch size for sending prompts. Defaults to 10.
                Note: If providing max requests per minute on the prompt_target, this should be set to 1 to
                ensure proper rate limit management.
            verbose (bool, Optional): If set to True, verbose output will be enabled. Defaults to False.
        """
        super().__init__(prompt_converters=prompt_converters, verbose=verbose)

        self._prompt_normalizer = PromptNormalizer()

        self._skeleton_key_prompt = (
            skeleton_key_prompt
            if skeleton_key_prompt
            else SeedPromptDataset.from_yaml_file(
                Path(DATASETS_PATH) / "orchestrators" / "skeleton_key" / "skeleton_key.prompt"
            )
            .prompts[0]
            .value
        )

        self._prompt_target = prompt_target

        self._batch_size = batch_size

    async def send_skeleton_key_with_prompt_async(
        self,
        *,
        prompt: str,
    ) -> PromptRequestResponse:
        """
        Sends a skeleton key, followed by the attack prompt to the target.

        Args

            prompt (str): The prompt to be sent.
            prompt_type (PromptDataType, Optional): The type of the prompt (e.g., "text"). Defaults to "text".

        Returns:
            PromptRequestResponse: The response from the prompt target.
        """

        conversation_id = str(uuid4())

        skeleton_key_prompt = SeedPromptGroup(prompts=[SeedPrompt(value=self._skeleton_key_prompt, data_type="text")])

        converter_configuration = PromptConverterConfiguration(converters=self._prompt_converters)

        await self._prompt_normalizer.send_prompt_async(
            seed_prompt_group=skeleton_key_prompt,
            conversation_id=conversation_id,
            request_converter_configurations=[converter_configuration],
            target=self._prompt_target,
            labels=self._global_memory_labels,
            orchestrator_identifier=self.get_identifier(),
        )

        objective_prompt = SeedPromptGroup(prompts=[SeedPrompt(value=prompt, data_type="text")])

        return await self._prompt_normalizer.send_prompt_async(
            seed_prompt_group=objective_prompt,
            conversation_id=conversation_id,
            request_converter_configurations=[converter_configuration],
            target=self._prompt_target,
            labels=self._global_memory_labels,
            orchestrator_identifier=self.get_identifier(),
        )

    async def send_skeleton_key_with_prompts_async(
        self,
        *,
        prompt_list: list[str],
    ) -> list[PromptRequestResponse]:
        """
        Sends a skeleton key and prompt to the target for each prompt in a list of prompts.

        Args:
            prompt_list (list[str]): The list of prompts to be sent.
            prompt_type (PromptDataType, Optional): The type of the prompts (e.g., "text"). Defaults to "text".

        Returns:
            list[PromptRequestResponse]: The responses from the prompt target.
        """

        return await batch_task_async(
            task_func=self.send_skeleton_key_with_prompt_async,
            task_arguments=["prompt"],
            prompt_target=self._prompt_target,
            batch_size=self._batch_size,
            items_to_batch=[prompt_list],
        )

    def print_conversation(self) -> None:
        """Prints all the conversations that have occured with the prompt target."""

        target_messages = self.get_memory()

        if not target_messages or len(target_messages) == 0:
            print("No conversation with the target")
            return

        for message in target_messages:
            if message.role == "user":
                print(f"{Style.BRIGHT}{Fore.RED}{message.role}: {message.converted_value}\n")
            else:
                print(f"{Style.BRIGHT}{Fore.GREEN}{message.role}: {message.converted_value}\n")
