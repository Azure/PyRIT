# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
from pathlib import Path
from typing import Optional
from uuid import uuid4

from colorama import Fore, Style

from pyrit.common.batch_helper import batch_task_async
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
from pyrit.orchestrator.single_turn.prompt_sending_orchestrator import PromptSendingOrchestrator

logger = logging.getLogger(__name__)


class SkeletonKeyOrchestrator(PromptSendingOrchestrator):
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
        super().__init__(
            objective_target=prompt_target,
            prompt_converters=prompt_converters,
            batch_size=batch_size,
            verbose=verbose
        )

        self._skeleton_key_prompt = (
            skeleton_key_prompt
            if skeleton_key_prompt
            else SeedPromptDataset.from_yaml_file(
                Path(DATASETS_PATH) / "orchestrators" / "skeleton_key" / "skeleton_key.prompt"
            )
            .prompts[0]
            .value
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

        Returns:
            list[PromptRequestResponse]: The responses from the prompt target.
        """
        if hasattr(self._prompt_target, 'rpm') and self._prompt_target.rpm and self._batch_size != 1:
            raise ValueError(
                "When using a prompt target with max_requests_per_minute, batch_size must be set to 1"
            )

        # Create a single conversation ID for the entire sequence
        conversation_id = str(uuid4())
        metadata = {"conversation_id": conversation_id}

        # First, send all skeleton keys
        skeleton_keys = [self._skeleton_key_prompt] * len(prompt_list)
        await self.send_prompts_async(
            prompt_list=skeleton_keys,
            metadata=metadata
        )

        # Then send all attack prompts with the same conversation ID
        attack_responses = await self.send_prompts_async(
            prompt_list=prompt_list,
            metadata=metadata
        )
        
        return attack_responses

    async def send_skeleton_key_with_prompt_async(
        self,
        *,
        prompt: str,
    ) -> PromptRequestResponse:
        """
        Sends a skeleton key, followed by the attack prompt to the target.

        Args:
            prompt (str): The prompt to be sent.

        Returns:
            PromptRequestResponse: The response from the prompt target.
        """
        # Create a single conversation ID
        conversation_id = str(uuid4())
        metadata = {"conversation_id": conversation_id}

        # Create normalizer requests for both prompts
        skeleton_request = self._create_normalizer_request(
            prompt_text=self._skeleton_key_prompt,
            prompt_type="text",
            converters=self._prompt_converters,
            metadata=metadata,
            conversation_id=conversation_id,
        )

        attack_request = self._create_normalizer_request(
            prompt_text=prompt,
            prompt_type="text",
            converters=self._prompt_converters,
            metadata=metadata,
            conversation_id=conversation_id,
        )

        # Send both requests in a single batch
        responses = await self.send_normalizer_requests_async(
            prompt_request_list=[skeleton_request, attack_request]
        )
        
        # Return the attack prompt response (second response)
        return responses[1]

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