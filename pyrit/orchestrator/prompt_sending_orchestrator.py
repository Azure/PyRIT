# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from collections import defaultdict
import uuid
from colorama import Fore, Style
import logging
from dataclasses import dataclass

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
        # Set the scorer and scorer._prompt_target memory to match the orchestrator's memory.
        if self._scorers:
            for scorer in self._scorers:
                scorer._memory = self._memory
                if hasattr(scorer, "_prompt_target"):
                    scorer._prompt_target._memory = self._memory

        self._prompt_target = prompt_target
        self._prompt_target._memory = self._memory

        self._batch_size = batch_size
        self._prepended_conversation: list[PromptRequestResponse] = None

    def set_prepended_conversation(self, *, prepended_conversation: list[PromptRequestResponse]):
        """
        Prepends a conversation to the prompt target.
        """
        self._prepended_conversation = prepended_conversation

    async def send_prompts_async(
        self,
        *,
        prompt_list: list[str],
        prompt_type: PromptDataType = "text",
        memory_labels: Optional[dict[str, str]] = None,
        metadata: Optional[str] = None,
        retry_on_false_scorer: Scorer = None,
        max_retry_on_false_scorer: int = 3,
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
                    metadata=metadata,
                )
            )

        return await self.send_normalizer_requests_async(
            prompt_request_list=requests,
            memory_labels=memory_labels,
            retry_on_false_scorer=retry_on_false_scorer,
            max_retry_on_false_scorer=max_retry_on_false_scorer,
        )

    async def send_normalizer_requests_async(
        self,
        *,
        prompt_request_list: list[NormalizerRequest],
        memory_labels: Optional[dict[str, str]] = None,
        retry_on_false_scorer: Scorer = None,
        max_retry_on_false_scorer: int = 3,
    ) -> list[PromptRequestResponse]:
        """
        Sends the normalized prompts to the prompt target.
        TODO: add documentation describing retry logic
        """
        for request in prompt_request_list:
            request.validate()

        conversation_id = self._prepare_conversation()

        for prompt in prompt_request_list:
            prompt.conversation_id = conversation_id

        # Normalizer is responsible for storing the requests in memory
        # The labels parameter may allow me to stash class information for each kind of prompt.
        responses: list[PromptRequestResponse] = await self._prompt_normalizer.send_prompt_batch_to_target_async(
            requests=prompt_request_list,
            target=self._prompt_target,
            labels=self._combine_with_global_memory_labels(memory_labels),
            orchestrator_identifier=self.get_identifier(),
            batch_size=self._batch_size,
        )

        if retry_on_false_scorer:
            retry_count = 0
            scores = await self._score_retry_async(
                prompt_list=prompt_request_list, response_list=responses, scorer=retry_on_false_scorer
            )
            retry_result_list = [
                RetryResult(prompt, response, score.get_value())
                for prompt, response, score in zip(prompt_request_list, responses, scores)
            ]

            while (
                any([retry_result.should_retry for retry_result in retry_result_list])
                and retry_count < max_retry_on_false_scorer
            ):
                print(
                    f"At least one response scored as 'should retry' (retry {retry_count+1}/{max_retry_on_false_scorer})"
                )
                retry_idx = [idx for idx, retry_result in enumerate(retry_result_list) if retry_result.should_retry]
                retry_prompts = [retry_result.prompt for retry_result in retry_result_list if retry_result.should_retry]

                retry_responses: list[PromptRequestResponse] = (
                    await self._prompt_normalizer.send_prompt_batch_to_target_async(
                        requests=retry_prompts,
                        target=self._prompt_target,
                        labels=self._combine_with_global_memory_labels(memory_labels),
                        orchestrator_identifier=self.get_identifier(),
                        batch_size=self._batch_size,
                    )
                )

                retry_scores = await self._score_retry_async(
                    prompt_list=retry_prompts, response_list=retry_responses, scorer=retry_on_false_scorer
                )
                for idx, response, score in zip(retry_idx, retry_responses, retry_scores):
                    retry_result_list[idx].response = response
                    retry_result_list[idx].should_retry = score.get_value()

                retry_count += 1

            responses = [retry_result.response for retry_result in retry_result_list]

        if self._scorers:
            response_ids = []
            for response in responses:
                for piece in response.request_pieces:
                    id = str(piece.id)
                    response_ids.append(id)
            await self._score_responses_async(response_ids)

        return responses

    async def _score_retry_async(
        self,
        prompt_list: list[NormalizerRequest],
        response_list: list[PromptRequestResponse],
        scorer: Scorer,
    ):
        first_prompt = prompt_list[0]
        text_idx = None
        for idx, prompt_piece in enumerate(first_prompt.request_pieces):
            if prompt_piece.prompt_data_type == "text":
                text_idx = idx
        if text_idx is not None:
            tasks = [p.request_pieces[text_idx].prompt_value for p in prompt_list]
        else:
            tasks = None
        response_pieces = [p.request_pieces[0] for p in response_list]
        scores = await scorer.score_prompts_batch_async(request_responses=response_pieces, tasks=tasks)
        return scores

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

    async def print_conversations(self):
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
                    await display_response(message, self._memory)

                scores = self._memory.get_scores_by_prompt_ids(prompt_request_response_ids=[message.id])
                for score in scores:
                    print(f"{Style.RESET_ALL}score: {score} : {score.score_rationale}")

    def _prepare_conversation(self):
        """
        Adds the conversation to memory if there is a prepended conversation, and return the conversation ID.
        """
        conversation_id = None
        if self._prepended_conversation:
            conversation_id = uuid.uuid4()
            for request in self._prepended_conversation:
                for piece in request.request_pieces:
                    piece.conversation_id = conversation_id

                self._memory.add_request_response_to_memory(request=request)
        return conversation_id


@dataclass
class RetryResult:
    prompt: NormalizerRequest
    response: PromptRequestResponse
    should_retry: bool
