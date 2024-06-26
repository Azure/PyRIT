# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging


from typing import Optional, List
from colorama import Style, Fore


from pyrit.memory import MemoryInterface
from pyrit.models.prompt_request_piece import PromptDataType
from pyrit.models.prompt_request_response import PromptRequestResponse
from pyrit.orchestrator import Orchestrator
from pyrit.prompt_normalizer import PromptNormalizer
from pyrit.prompt_normalizer.normalizer_request import NormalizerRequest
from pyrit.prompt_target import PromptTarget
from pyrit.prompt_converter import PromptConverter
from pyrit.datasets import DatasetFetcher


logger = logging.getLogger(__name__)


class PromptSendingOrchestrator(Orchestrator):
    """
    This orchestrator takes a set of prompts, converts them using the list of PromptConverters,
    and sends them to a target.
    """

    def __init__(
        self,
        prompt_target: PromptTarget,
        prompt_converters: Optional[list[PromptConverter]] = None,
        memory: MemoryInterface = None,
        batch_size: int = 10,
        verbose: bool = False,
    ) -> None:
        """
        Args:
            prompt_target (PromptTarget): The target for sending prompts.
            prompt_converters (list[PromptConverter], optional): List of prompt converters. These are stacked in
                                    the order they are provided. E.g. the output of converter1 is the input of
                                    converter2.
            memory (MemoryInterface, optional): The memory interface. Defaults to None.
            batch_size (int, optional): The (max) batch size for sending prompts. Defaults to 10.
        """
        super().__init__(prompt_converters=prompt_converters, memory=memory, verbose=verbose)

        self._prompt_normalizer = PromptNormalizer(memory=self._memory)

        self._prompt_target = prompt_target
        self._prompt_target._memory = self._memory

        self._batch_size = batch_size

    async def send_prompts_async(
        self, *, prompt_list: list[str], prompt_type: PromptDataType = "text"
    ) -> list[PromptRequestResponse]:
        """
        Sends the prompts to the prompt target.
        """

        requests: list[NormalizerRequest] = []
        for prompt in prompt_list:
            requests.append(
                self._create_normalizer_request(
                    prompt_text=prompt,
                    prompt_type=prompt_type,
                    converters=self._prompt_converters,
                )
            )

        for request in requests:
            request.validate()

        return await self._prompt_normalizer.send_prompt_batch_to_target_async(
            requests=requests,
            target=self._prompt_target,
            labels=self._global_memory_labels,
            orchestrator_identifier=self.get_identifier(),
            batch_size=self._batch_size,
        )

    async def send_normalizer_requests_async(
        self, *, prompt_request_list: list[NormalizerRequest]
    ) -> list[PromptRequestResponse]:
        """
        Sends the prompts to the prompt target.
        """
        for request in prompt_request_list:
            request.validate()

        return await self._prompt_normalizer.send_prompt_batch_to_target_async(
            requests=prompt_request_list,
            target=self._prompt_target,
            labels=self._global_memory_labels,
            orchestrator_identifier=self.get_identifier(),
            batch_size=self._batch_size,
        )
    
    
    def fetch_many_shot_examples(self, source: str, source_type: str = 'repository') -> list[dict]:
        """
        Fetch many-shot examples from a specified source.

        Args:
            source (str): The source from which to fetch examples.
            source_type (str): The type of source ('repository' or 'user'). Defaults to 'repository'.

        Returns:
            list[dict]: A list of examples.
        """
        dataset_fetcher = DatasetFetcher()
        return dataset_fetcher.import_examples(source=source, source_type=source_type)
    

    def print_conversation(self, responses: List[PromptRequestResponse]):
        """Prints the conversation between the prompt target and the user."""
        if not responses:
            print("No conversation with the target")
            return

        for response in responses:
            for piece in response.request_pieces:
                # Print the role, original value, and converted value assuming both are always present
                print(f"{Style.BRIGHT}{Fore.RED}{piece.role} (Original): {piece.original_value}\n")
                print(f"{Style.BRIGHT}{Fore.GREEN}{piece.role} (Converted): {piece.converted_value}\n") # Do we need to print the converted value?
    