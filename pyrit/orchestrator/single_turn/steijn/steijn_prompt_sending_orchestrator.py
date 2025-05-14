# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import asyncio
import logging
import re
import uuid
from typing import Optional, Union, List, Dict, Any

from colorama import Fore, Style

from pyrit.common.display_response import display_image_response
from pyrit.common.utils import combine_dict
from pyrit.models import PromptRequestResponse
from pyrit.models.filter_criteria import PromptConverterState, PromptFilterCriteria
from pyrit.orchestrator import Orchestrator
from pyrit.prompt_converter import PromptConverter
from pyrit.prompt_normalizer import NormalizerRequest, PromptNormalizer
from pyrit.prompt_target import PromptChatTarget, PromptTarget
from pyrit.score import Scorer

logger = logging.getLogger(__name__)


class SteijnPromptSendingOrchestrator(Orchestrator):
    """
    This orchestrator takes a set of prompts, converts them using the list of PromptConverters,
    sends them to a target, and scores the responses with scorers (if provided).

    It supports both single-turn and multi-turn conversational test cases.
    For multi-turn conversations, all turns share the same conversation ID.
    """

    def __init__(
            self,
            objective_target: PromptTarget,
            prompt_converters: Optional[List[PromptConverter]] = None,
            scorers: Optional[List[Scorer]] = None,
            batch_size: int = 1,
            verbose: bool = False,
    ) -> None:
        """
        Args:
            objective_target (PromptTarget): The target for sending prompts.
            prompt_converters (List[PromptConverter], Optional): List of prompt converters.
            scorers (List[Scorer], Optional): List of scorers for evaluating prompt responses.
            batch_size (int, Optional): The (max) batch size for sending prompts.
            verbose (bool, Optional): Enables verbose logging.
        """
        super().__init__(prompt_converters=prompt_converters, verbose=verbose)
        self._prompt_normalizer = PromptNormalizer()
        self._scorers = scorers or []
        self._objective_target = objective_target
        self._batch_size = batch_size
        self._prepended_conversation: Optional[List[PromptRequestResponse]] = None

    def set_prepended_conversation(self, *, prepended_conversation: List[PromptRequestResponse]):
        """
        Prepends a conversation to the prompt target.
        This is sent along with each prompt request and can be the first part of a conversation.
        """
        if prepended_conversation and not isinstance(self._objective_target, PromptChatTarget):
            raise TypeError(
                f"Only PromptChatTargets are able to modify conversation history. Instead objective_target is: "
                f"{type(self._objective_target)}."
            )
        self._prepended_conversation = prepended_conversation

    async def get_prepended_conversation_async(
            self, *, normalizer_request: NormalizerRequest
    ) -> Optional[List[PromptRequestResponse]]:
        """
        Returns the prepended conversation for the normalizer request.
        Can be overwritten by subclasses to provide a different conversation.
        """
        if self._prepended_conversation:
            return self._prepended_conversation
        return None

    def set_skip_criteria(
            self, *, skip_criteria: PromptFilterCriteria, skip_value_type: PromptConverterState = "original"
    ):
        """
        Sets the skip criteria for the orchestrator.
        If prompts match this in memory, they won't be sent to a target.
        """
        self._prompt_normalizer.set_skip_criteria(skip_criteria=skip_criteria, skip_value_type=skip_value_type)

    # Todo: This function will be refactored into run_attacks_async
    # Todo: Instead of Normalizer request object, Objective, Expected output, and prompt type should be passed.
    async def send_normalizer_requests_async(
            self,
            *,
            prompt_request_list: List[NormalizerRequest],
            memory_labels: Optional[Dict[str, str]] = None,
    ) -> List[PromptRequestResponse]:
        """
        Sends the normalized prompts to the prompt target.
        """
        expected_output_list = []
        request_prompts = []
        self.validate_normalizer_requests(prompt_request_list=prompt_request_list)

        for prompt in prompt_request_list:
            # Reuse conversation_id if already set; otherwise, generate a new one.
            prompt.conversation_id = await self._prepare_conversation_async(normalizer_request=prompt)
            request_prompts.append(prompt.seed_prompt_group.prompts[0].value)
            if prompt.seed_prompt_group.prompts[0].expected_output:
                expected_output_list.append(prompt.seed_prompt_group.prompts[0].expected_output)

        # Todo: In the new design, do not forget to set the expected output and reference prompt before sending the request to scorer.
        responses: List[PromptRequestResponse] = await self._prompt_normalizer.send_prompt_batch_to_target_async(
            requests=prompt_request_list,
            target=self._objective_target,
            labels=combine_dict(existing_dict=self._global_memory_labels, new_dict=memory_labels),
            orchestrator_identifier=self.get_identifier(),
            batch_size=self._batch_size,
        )

        response_pieces = []
        if self._scorers and responses:
            response_pieces = PromptRequestResponse.flatten_to_prompt_request_pieces(responses)
            for i, piece in enumerate(response_pieces):
                if i < len(expected_output_list):
                    piece.expected_output = expected_output_list[i]
                if i < len(request_prompts):
                    piece.prompt_metadata["reference_prompt"] = request_prompts[i]

        for scorer in self._scorers:
            await scorer.score_responses_inferring_tasks_batch_async(
                request_responses=response_pieces, batch_size=5
            )

        return responses

    async def send_qa_pairs_async(self, qa_pairs: List[Dict[str, Any]]) -> list[PromptRequestResponse]:
        """
        Sends a list of QA pairs to the prompt target.
        Supports both single-turn and multi-turn conversational test cases.
        For multi-turn cases, all turns in a conversation share the same conversation ID.
        For multi-turn conversations, the first turn's response is awaited so that its thread ID can be extracted
        and then the target's HTTP request URL is updated accordingly.
        Single-turn cases are batched together.
        """
        all_responses = []
        single_turn_requests: List[NormalizerRequest] = []
        start_request_copy = self._objective_target.http_request

        for i, qa in enumerate(qa_pairs):
            print("\nExecuting test case:", i+1)
            self._objective_target.http_request = start_request_copy # Reset to the original request for each test case.

            # Multi-turn test case.
            if "conversation" in qa:
                # Flush any accumulated single-turn requests.
                if single_turn_requests:
                    # Todo: This function will be refactored into run_attacks_async
                    await self.send_normalizer_requests_async(prompt_request_list=single_turn_requests)
                    single_turn_requests = []

                conversation_id = str(uuid.uuid4())
                is_thread_id_set = False
                for idx, turn in enumerate(qa["conversation"]):
                    prompt_text = turn["question"]
                    print("Question:", prompt_text)
                    expected_output = turn["expected_outcome"]

                    # Todo: We can pass these information directly to run attacks function
                    request = self._create_normalizer_request(
                        prompt_text=prompt_text,
                        expected_output=expected_output,
                        prompt_type="text",
                        converters=self._prompt_converters,
                        metadata=None,
                        conversation_id=conversation_id,
                    )

                    # Todo: This function will be refactored into run_attacks_async
                    results = await self.send_normalizer_requests_async(prompt_request_list=[request])
                    all_responses.extend(results)
                    # Todo: OrchestratorResult will be returned, find a way to get the thread_id from the result. You can use metadata
                    flattened = PromptRequestResponse.flatten_to_prompt_request_pieces(results)
                    if idx == 0:
                        thread_id = flattened[0].prompt_metadata.get("thread_id")
                        if thread_id and not is_thread_id_set:
                            # Update the target's HTTP URL to include the threadId.
                            match = re.search(r"(https?://[^\s/$.?#].[^\s]*)", self._objective_target.http_request)
                            if match:
                                url = match.group(1)
                                self._objective_target.http_request = self._objective_target.http_request.replace(
                                    url, f"{url}?threadId={thread_id}"
                                )
                                is_thread_id_set = True
                        else:
                            print("Thread ID not found in the first turn's response. Aborting this conversation.")
                            break

                    # Optionally, wait a bit between turns.
                    await asyncio.sleep(1)
            else:
                # Single-turn test case: accumulate the request.
                prompt_text = qa["question"]
                print("Question:", prompt_text)
                expected_output = qa["expected_outcome"]
                request = self._create_normalizer_request(
                    prompt_text=prompt_text,
                    expected_output=expected_output,
                    prompt_type="text",
                    converters=self._prompt_converters,
                    metadata=None,
                    conversation_id=str(uuid.uuid4()),
                )
                single_turn_requests.append(request)

        # Flush any remaining single-turn requests in one batch.
        if single_turn_requests:
            results = await self.send_normalizer_requests_async(prompt_request_list=single_turn_requests)
            all_responses.extend(results)

        return all_responses

    def validate_normalizer_requests(self, *, prompt_request_list: List[NormalizerRequest]):
        """
        Validates the normalizer requests.
        This is a no-op for this orchestrator, but subclasses may implement additional checks.
        """
        pass

    async def _prepare_conversation_async(self, normalizer_request: NormalizerRequest) -> str:
        """
        Adds the conversation to memory if there is a prepended conversation, and returns the conversation ID.
        If a conversation ID is already provided in the NormalizerRequest, it is reused.
        """
        conversation_id = normalizer_request.conversation_id or str(uuid.uuid4())
        prepended_conversation = await self.get_prepended_conversation_async(normalizer_request=normalizer_request)
        if prepended_conversation:
            for request in prepended_conversation:
                for piece in request.request_pieces:
                    piece.conversation_id = conversation_id
                    piece.orchestrator_identifier = self.get_identifier()
                    piece.id = uuid.uuid4()
                self._memory.add_request_response_to_memory(request=request)
        return conversation_id

    def get_all_chat_results(self) -> List[dict]:
        """
        Retrieves all chat results from the orchestrator's memory by grouping messages by conversation ID.

        For each conversation:
          - If the conversation contains exactly one user message followed by one assistant message,
            it returns a simplified dictionary with keys "prompt", "assistant_response", and "scores".
          - Otherwise, it returns the full transcript under the key "conversation".

        Returns:
            List[dict]: A list of conversation results.
        """
        messages = self.get_memory()
        conv_dict: Dict[str, List[Dict[str, Any]]] = {}

        # Group messages by conversation_id.
        for msg in messages:
            conv_id = msg.conversation_id
            if conv_id not in conv_dict:
                conv_dict[conv_id] = []
            entry = {
                "role": msg.role,
                "message": msg.converted_value
            }
            if msg.scores:
                entry["scores"] = [
                    {
                        "score_value": s.score_value,
                        "score_rationale": s.score_rationale,
                        "expected_output": s.expected_output
                    }
                    for s in msg.scores
                ]
            conv_dict[conv_id].append(entry)

        results = []
        for conv_id, conversation in conv_dict.items():
            # If conversation has exactly one user and one assistant message, return a pair structure.
            if (len(conversation) == 2 and
                    conversation[0]["role"].lower() == "user" and
                    conversation[1]["role"].lower() == "assistant"):
                results.append({
                    "conversation_id": conv_id,
                    "prompt": conversation[0]["message"],
                    "assistant_response": conversation[1]["message"],
                    "scores": conversation[1].get("scores", [])
                })
            else:
                # Otherwise, return the entire conversation transcript.
                results.append({
                    "conversation_id": conv_id,
                    "conversation": conversation
                })
        return results

