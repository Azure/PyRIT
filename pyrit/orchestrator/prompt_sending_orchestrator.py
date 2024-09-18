# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from collections import defaultdict
from colorama import Fore, Style
import logging

from typing import Optional

from pyrit.common.display_response import display_response
from pyrit.memory import MemoryInterface
from pyrit.models import PromptDataType
from pyrit.models import PromptRequestResponse
from pyrit.orchestrator import Orchestrator
from pyrit.orchestrator.scoring_orchestrator import ScoringOrchestrator
from pyrit.prompt_normalizer import PromptNormalizer
from pyrit.prompt_normalizer.normalizer_request import NormalizerRequest
from pyrit.prompt_target import PromptTarget
from pyrit.prompt_converter import PromptConverter
from pyrit.score import Scorer


logger = logging.getLogger(__name__)


class PromptSendingOrchestrator(Orchestrator):
    """
    This orchestrator takes a set of prompts, converts them using the list of PromptConverters,
    sends them to a target, and scores the resonses with scorers (if provided).
    """

    def __init__(
        self,
        prompt_target: PromptTarget,
        prompt_converters: Optional[list[PromptConverter]] = None,
        scorers: Optional[list[Scorer]] = None,
        memory: MemoryInterface = None,
        memory_labels: Optional[dict[str, str]] = None,
        batch_size: int = 10,
        verbose: bool = False,
    ) -> None:
        """
        Args:
            prompt_target (PromptTarget): The target for sending prompts.
            prompt_converters (list[PromptConverter], optional): List of prompt converters. These are stacked in
                the order they are provided. E.g. the output of converter1 is the input of converter2.
            scorers (list[Scorer], optional): List of scorers to use for each prompt request response, to be
                scored immediately after receiving response. Default is None.
            memory (MemoryInterface, optional): The memory interface. Defaults to None.
            memory_labels (dict[str, str], optional): A free-form dictionary for tagging prompts with custom labels.
            These labels can be used to track all prompts sent as part of an operation, score prompts based on
            the operation ID (op_id), and tag each prompt with the relevant Responsible AI (RAI) harm category.
            Users can define any key-value pairs according to their needs. Defaults to None.
            batch_size (int, optional): The (max) batch size for sending prompts. Defaults to 10.
                Note: If providing max requests per minute on the prompt_target, this should be set to 1 to
                ensure proper rate limit management.
        """
        super().__init__(
            prompt_converters=prompt_converters, memory=memory, memory_labels=memory_labels, verbose=verbose
        )

        self._prompt_normalizer = PromptNormalizer(memory=self._memory)
        self._scorers = scorers

        self._prompt_target = prompt_target
        self._prompt_target._memory = self._memory

        self._batch_size = batch_size

    async def send_prompts_async(
        self,
        *,
        prompt_list: list[str],
        prompt_type: PromptDataType = "text",
        memory_labels: Optional[dict[str, str]] = None,
        metadata: Optional[str] = None,
    ) -> list[PromptRequestResponse]:
        """
        Sends the prompts to the prompt target.

        Args:
            prompt_list (list[str]): The list of prompts to be sent.
            prompt_type (PromptDataType): The type of prompt data. Defaults to "text".
            memory_labels (dict[str, str], optional): A free-form dictionary of additional labels to apply to the
                prompts.
            These labels will be merged with the instance's global memory labels. Defaults to None.
            metadata: Any additional information to be added to the memory entry corresponding to the prompts sent.

        Returns:
            list[PromptRequestResponse]: The responses from sending the prompts.
        """

        if isinstance(prompt_list, str):
            prompt_list = [prompt_list]

        requests: list[NormalizerRequest] = []
        for prompt in prompt_list:
            requests.append(
                self._create_normalizer_request(
                    prompt_text=prompt,
                    prompt_type=prompt_type,
                    converters=self._prompt_converters,
                    metadata=metadata if metadata else None,  # NOTE added
                )
            )

        return await self.send_normalizer_requests_async(
            prompt_request_list=requests,
            memory_labels=memory_labels,
        )

    async def send_prompt_async(
        self,
        *,
        prompt: str,
        prompt_type: PromptDataType = "text",
        memory_labels: Optional[dict[str, str]] = None,
        conversation_id: Optional[str] = None,
    ) -> PromptRequestResponse:
        """
        Sends a single prompts to the prompt target. Can be used for multi-turn using conversation_id.

        Args:
            prompt (list[str]): The prompt to be sent.
            prompt_type (PromptDataType): The type of prompt data. Defaults to "text".
            memory_labels (dict[str, str], optional): A free-form dictionary of extra labels to apply to the prompts.
                These labels will be merged with the instance's global memory labels. Defaults to None.
            conversation_id (str, optional): The conversation ID to use for multi-turn conversation. Defaults to None.

        Returns:
            list[PromptRequestResponse]: The responses from sending the prompts.
        """

        normalizer_request = self._create_normalizer_request(
            prompt_text=prompt,
            prompt_type=prompt_type,
            converters=self._prompt_converters,
        )

        return await self._prompt_normalizer.send_prompt_async(
            normalizer_request=normalizer_request,
            target=self._prompt_target,
            conversation_id=conversation_id,
            labels=self._combine_with_global_memory_labels(memory_labels),
            orchestrator_identifier=self.get_identifier(),
        )

    async def send_normalizer_requests_async(
        self,
        *,
        prompt_request_list: list[NormalizerRequest],
        memory_labels: Optional[dict[str, str]] = None,
    ) -> list[PromptRequestResponse]:
        """
        Sends the normalized prompts to the prompt target.
        """
        for request in prompt_request_list:
            request.validate()

        # Normalizer is responsible for storing the requests in memory
        # The labels parameter may allow me to stash class information for each kind of prompt.
        responses: list[PromptRequestResponse] = await self._prompt_normalizer.send_prompt_batch_to_target_async(
            requests=prompt_request_list,
            target=self._prompt_target,
            labels=self._combine_with_global_memory_labels(memory_labels),
            orchestrator_identifier=self.get_identifier(),
            batch_size=self._batch_size,
        )

        # These are the responses from the target
        # print(responses[0].request_pieces)

        if self._scorers:
            response_ids = []
            for response in responses:
                for piece in response.request_pieces:
                    id = str(piece.id)
                    response_ids.append(id)

            await self._score_responses_async(response_ids)

        return responses

    async def _score_responses_async(self, prompt_ids: list[str]):
        with ScoringOrchestrator(
            memory=self._memory,
            batch_size=self._batch_size,
            verbose=self._verbose,
        ) as scoring_orchestrator:
            for scorer in self._scorers:
                await scoring_orchestrator.score_prompts_by_request_id_async(
                    scorer=scorer,
                    prompt_ids=prompt_ids,
                    responses_only=True,
                )

    def print_conversations(self):
        """Prints the conversation between the prompt target and the red teaming bot."""
        all_messages = self.get_memory()

        # group by conversation ID
        messages_by_conversation_id = defaultdict(list)
        for message in all_messages:
            messages_by_conversation_id[message.conversation_id].append(message)

        for conversation_id in messages_by_conversation_id:
            messages_by_conversation_id[conversation_id].sort(key=lambda x: x.sequence)

            print(f"{Style.NORMAL}{Fore.RESET}Conversation ID: {conversation_id}")

            if not messages_by_conversation_id[conversation_id]:
                print("No conversation with the target")
                continue

            for message in messages_by_conversation_id[conversation_id]:
                if message.role == "user":
                    print(f"{Style.BRIGHT}{Fore.BLUE}{message.role}: {message.converted_value}")
                else:
                    print(f"{Style.NORMAL}{Fore.YELLOW}{message.role}: {message.converted_value}")
                    display_response(message)

                scores = self._memory.get_scores_by_prompt_ids(prompt_request_response_ids=[message.id])
                for score in scores:
                    print(f"{Style.RESET_ALL}score: {score} : {score.score_rationale}")

    def _combine_with_global_memory_labels(self, memory_labels: dict[str, str]) -> dict[str, str]:
        """
        Combines the global memory labels with the provided memory labels.
        The passed memory_leabels take prcedence with collisions.
        """
        return {**(self._global_memory_labels or {}), **(memory_labels or {})}
