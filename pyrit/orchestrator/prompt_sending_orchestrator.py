# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from collections import defaultdict
import uuid
from colorama import Fore, Style
import logging

from typing import Optional
from dataclasses import dataclass

from pyrit.common.display_response import display_image_response
from pyrit.models import PromptDataType
from pyrit.models import PromptRequestResponse, Score
from pyrit.orchestrator import Orchestrator
from pyrit.orchestrator.scoring_orchestrator import ScoringOrchestrator
from pyrit.prompt_normalizer import PromptNormalizer
from pyrit.prompt_normalizer.normalizer_request import NormalizerRequest
from pyrit.prompt_target import PromptTarget, OpenAIChatTarget
from pyrit.prompt_converter import PromptConverter
from pyrit.score import Scorer, SelfAskRefusalScorer


logger = logging.getLogger(__name__)


@dataclass
class RetryResult:
    "Class to keep track of whether a prompt/response pair should be retried."
    prompt: NormalizerRequest
    response: PromptRequestResponse = None
    should_retry: bool = True


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
        memory_labels: Optional[dict[str, str]] = None,
        batch_size: int = 10,
        verbose: bool = False,
        retry_scorer: Scorer = None,
    ) -> None:
        """
        Args:
            prompt_target (PromptTarget): The target for sending prompts.
            prompt_converters (list[PromptConverter], Optional): List of prompt converters. These are stacked in
                the order they are provided. E.g. the output of converter1 is the input of converter2.
            scorers (list[Scorer], Optional): List of scorers to use for each prompt request response, to be
                scored immediately after receiving response. Default is None.
            memory_labels (dict[str, str], Optional): A free-form dictionary for tagging prompts with custom labels.
            These labels can be used to track all prompts sent as part of an operation, score prompts based on
            the operation ID (op_id), and tag each prompt with the relevant Responsible AI (RAI) harm category.
            Users can define any key-value pairs according to their needs. Defaults to None.
            batch_size (int, Optional): The (max) batch size for sending prompts. Defaults to 10.
                Note: If providing max requests per minute on the prompt_target, this should be set to 1 to
                ensure proper rate limit management.
            retry_scorer (Scorer, optional): The Scorer used to optionally determine whether a prompt should be
                retried.
        """
        super().__init__(prompt_converters=prompt_converters, memory_labels=memory_labels, verbose=verbose)

        self._prompt_normalizer = PromptNormalizer()
        self._scorers = scorers

        if retry_scorer is not None:
            if retry_scorer.scorer_type != "true_false":
                raise TypeError("retry_scorer must be a 'true_false' scorer")
            self._retry_scorer = retry_scorer
        else:
            self._retry_scorer = SelfAskRefusalScorer(chat_target=OpenAIChatTarget())

        self._prompt_target = prompt_target

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
        max_retries_on_refusal: int = 0,
    ) -> list[PromptRequestResponse]:
        """
        Sends the prompts to the prompt target.

        Args:
            prompt_list (list[str]): The list of prompts to be sent.
            prompt_type (PromptDataType): The type of prompt data. Defaults to "text".
            memory_labels (dict[str, str], Optional): A free-form dictionary of additional labels to apply to the
                prompts.
            These labels will be merged with the instance's global memory labels. Defaults to None.
            metadata: Any additional information to be added to the memory entry corresponding to the prompts sent.
            max_retries_on_refusal (int): The maximum number of times to re-send prompts while at least one response is
                scored as a refusal. Defaults to 0.

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
            max_retries_on_refusal=max_retries_on_refusal,
        )

    async def send_normalizer_requests_async(
        self,
        *,
        prompt_request_list: list[NormalizerRequest],
        memory_labels: Optional[dict[str, str]] = None,
        max_retries_on_refusal: int = 0,
    ) -> list[PromptRequestResponse]:
        """
        Sends the normalized prompts to the prompt target. Optionally re-sends the normalized prompts to the target
        while at least one response is scored as a refusal, up to a maximum number of retries.
        """
        for request in prompt_request_list:
            request.validate()

        conversation_id = self._prepare_conversation()

        for prompt in prompt_request_list:
            prompt.conversation_id = conversation_id

        responses = await self._handle_retries_with_refusal(
            prompt_request_list=prompt_request_list,
            memory_labels=memory_labels,
            max_retries_on_refusal=max_retries_on_refusal,
        )

        if self._scorers:
            response_ids = []
            for response in responses:
                for piece in response.request_pieces:
                    id = str(piece.id)
                    response_ids.append(id)
            await self._score_final_responses_async(response_ids)

        return responses

    async def _handle_retries_with_refusal(
        self,
        *,
        prompt_request_list: list[NormalizerRequest],
        memory_labels: Optional[dict[str, str]] = None,
        max_retries_on_refusal: int = 0,
    ) -> list[PromptRequestResponse]:
        """
        Helper function to handle the retry logic for PromptSendingOrchestrator. By default, the function
        gets responses only once (no retries). Otherwise, a SelfAskRefusalScorer is used to determine whether
        to re-send prompts to the target, up to a maximum number of retries.
        """

        # Initialize retry count and list of results
        retry_count = 0
        retry_result_list = [RetryResult(prompt=prompt) for prompt in prompt_request_list]

        while (
            any([retry_result.should_retry for retry_result in retry_result_list])
            and retry_count <= max_retries_on_refusal
        ):
            idx_list = [idx for idx, retry_result in enumerate(retry_result_list) if retry_result.should_retry]
            prompt_request_list = [
                retry_result.prompt for retry_result in retry_result_list if retry_result.should_retry
            ]

            # Normalizer is responsible for storing the requests in memory
            # The labels parameter may allow me to stash class information for each kind of prompt.
            responses: list[PromptRequestResponse] = await self._prompt_normalizer.send_prompt_batch_to_target_async(
                requests=prompt_request_list,
                target=self._prompt_target,
                labels=self._combine_with_global_memory_labels(memory_labels),
                orchestrator_identifier=self.get_identifier(),
                batch_size=self._batch_size,
            )
            for idx, response in zip(idx_list, responses):
                retry_result_list[idx].response = response

            # If max retries is zero, we don't need to score for retries and should break
            if max_retries_on_refusal == 0:
                break

            retry_scores = await self._get_should_retry_score_async(
                prompt_list=prompt_request_list,
                response_list=responses,
            )
            for idx, response, retry_score in zip(idx_list, responses, retry_scores):
                retry_result_list[idx].should_retry = retry_score.get_value()

            retry_count += 1

        responses = [retry_result.response for retry_result in retry_result_list]
        return responses

    async def _get_should_retry_score_async(
        self,
        prompt_list: list[NormalizerRequest],
        response_list: list[PromptRequestResponse],
    ) -> list[Score]:
        """
        Helper function that scores a list of responses to determine whether they should be re-sent to the prompt
        target. The responses are scored with a SelfAskRefusalScorer, which uses the original prompt to determine
        whether the response is a refusal.
        """
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

        scores = await self._retry_scorer.score_prompts_batch_async(request_responses=response_pieces, tasks=tasks)
        return scores

    async def _score_final_responses_async(self, prompt_ids: list[str]):
        """Helper function to score the final responses using a list of scorers."""
        with ScoringOrchestrator(
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
                    await display_image_response(message)

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

                    # if the piece is retrieved from somewhere else, it needs to be unique
                    # and if not, this won't hurt anything
                    piece.id = uuid.uuid4()

                self._memory.add_request_response_to_memory(request=request)
        return conversation_id
