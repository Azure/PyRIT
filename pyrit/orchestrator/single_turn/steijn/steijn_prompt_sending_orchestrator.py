# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import re
import asyncio
import logging
import uuid
from typing import Any, Optional, Sequence, List, Dict, Coroutine

from pyrit.common.utils import combine_dict
from pyrit.models import (
    PromptRequestResponse,
    SeedPrompt,
    SeedPromptGroup,
)
from pyrit.models.filter_criteria import PromptConverterState, PromptFilterCriteria
from pyrit.orchestrator import (
    Orchestrator,
    OrchestratorResult,
    OrchestratorResultStatus,
)
from pyrit.orchestrator.models.orchestrator_result import OrchestratorResult
from pyrit.prompt_normalizer import PromptConverterConfiguration, PromptNormalizer
from pyrit.prompt_target import PromptChatTarget, PromptTarget
from pyrit.prompt_target.batch_helper import batch_task_async
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
            request_converter_configurations: Optional[list[PromptConverterConfiguration]] = None,
            response_converter_configurations: Optional[list[PromptConverterConfiguration]] = None,
            objective_scorer: Optional[Scorer] = None,
            auxiliary_scorers: Optional[list[Scorer]] = None,
            should_convert_prepended_conversation: bool = True,
            batch_size: int = 10,
            retries_on_objective_failure: int = 0,
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
        super().__init__(verbose=verbose)

        self._prompt_normalizer = PromptNormalizer()

        if not objective_scorer:
            raise ValueError("Objective scorer must be provided.")

        self._objective_scorer = objective_scorer or None
        self._auxiliary_scorers = auxiliary_scorers or []

        self._objective_target = objective_target

        self._request_converter_configurations = request_converter_configurations or []
        self._response_converter_configurations = response_converter_configurations or []

        self._should_convert_prepended_conversation = should_convert_prepended_conversation
        self._batch_size = batch_size
        self._retries_on_objective_failure = retries_on_objective_failure

    def set_skip_criteria(
            self, *, skip_criteria: PromptFilterCriteria, skip_value_type: PromptConverterState = "original"
    ):
        """
        Sets the skip criteria for the orchestrator.
        If prompts match this in memory, they won't be sent to a target.
        """
        self._prompt_normalizer.set_skip_criteria(skip_criteria=skip_criteria, skip_value_type=skip_value_type)

    async def _add_prepended_conversation_to_memory(
            self,
            prepended_conversation: Optional[list[PromptRequestResponse]],
            conversation_id: str,
    ):
        """
        Processes the prepended conversation by converting it if needed and adding it to memory.

        Args:
            prepended_conversation (Optional[list[PromptRequestResponse]]): The conversation to prepend
            conversation_id (str): The conversation ID to use for the request pieces
        """
        if not prepended_conversation:
            return

        if not isinstance(self._objective_target, PromptChatTarget):
            raise ValueError("Prepended conversation can only be used with a PromptChatTarget")

        await self._prompt_normalizer.add_prepended_conversation_to_memory(
            prepended_conversation=prepended_conversation,
            conversation_id=conversation_id,
            should_convert=self._should_convert_prepended_conversation,
            converter_configurations=self._request_converter_configurations,
            orchestrator_identifier=self.get_identifier(),
        )

    async def _score_auxiliary_async(self, result: PromptRequestResponse) -> None:
        """
        Scores the response using auxiliary scorers if they are configured.

        Args:
            result (PromptRequestResponse): The response to score
        """
        if not self._auxiliary_scorers:
            return

        tasks = []
        for piece in result.request_pieces:
            if piece.role == "assistant":
                for scorer in self._auxiliary_scorers:
                    tasks.append(scorer.score_async(request_response=piece))

        if tasks:
            await asyncio.gather(*tasks)

    async def _score_objective_async(
            self, result: PromptRequestResponse, objective: str
    ) -> tuple[OrchestratorResultStatus, Optional[Any]]:
        """
        Scores the response using the objective scorer if configured.

        Args:
            result (PromptRequestResponse): The response to score
            objective (str): The objective to score against

        Returns:
            tuple[OrchestratorResultStatus, Optional[Any]]: A tuple containing the status and objective score
            If the objective_scorer returns a list of scores, the first score that is true will be returned as the
            objective score.
            Note, this behavior can be overridden by setting the objective_scorer to a CompositeScorer.
        """
        if not self._objective_scorer:
            return "unknown", None

        status: OrchestratorResultStatus = "failure"
        objective_score = None
        first_failure_score = None

        for piece in result.request_pieces:
            if piece.role == "assistant":
                objective_score_list = await self._objective_scorer.score_async(
                    request_response=piece,
                    task=objective,
                )

                # Find and save the first score that is true
                for score in objective_score_list:
                    if score.get_value():
                        objective_score = score
                        if float(score.score_value) > 0.5:
                            status = "success"
                        break
                    elif first_failure_score is None:
                        first_failure_score = score
                if status == "success":
                    break

        # If no success was found, use the first failure score
        if status == "failure" and first_failure_score is not None:
            objective_score = first_failure_score

        return status, objective_score

    async def execute_step_async(
            self,
            *,
            objective: str,
            expected_output: Optional[str] = None,
            seed_prompt: SeedPromptGroup = None,
            prepended_conversation: Optional[list[PromptRequestResponse]] = None,
            memory_labels: Optional[dict[str, str]] = None,
            conversation_id: str = ""
    ) -> tuple[None, None] | tuple[OrchestratorResult, PromptRequestResponse]:
        """
        Runs the attack and returns both the OrchestratorResult and the PromptRequestResponse.

        Returns:
            Tuple of (OrchestratorResult, PromptRequestResponse)
        """

        prompt_request_response = None
        if not seed_prompt:
            seed_prompt = SeedPromptGroup(prompts=[
                SeedPrompt(value=objective, expected_output=expected_output, data_type="text")
            ])

        status: OrchestratorResultStatus = "unknown"
        objective_score = None

        for _ in range(self._retries_on_objective_failure + 1):
            if conversation_id is None or conversation_id == "":
                conversation_id = str(uuid.uuid4())

            await self._add_prepended_conversation_to_memory(prepended_conversation, conversation_id)

            prompt_request_response = await self._prompt_normalizer.send_prompt_async(
                seed_prompt_group=seed_prompt,
                target=self._objective_target,
                conversation_id=conversation_id,
                request_converter_configurations=self._request_converter_configurations,
                response_converter_configurations=self._response_converter_configurations,
                labels=combine_dict(existing_dict=self._global_memory_labels, new_dict=memory_labels),
                orchestrator_identifier=self.get_identifier(),
            )

            if not prompt_request_response:
                return None, None

            piece = prompt_request_response.request_pieces[0]
            piece.expected_output = expected_output
            prompt_request_response.request_pieces = [piece]

            await self._score_auxiliary_async(prompt_request_response)
            status, objective_score = await self._score_objective_async(prompt_request_response, objective)

            if status == "success":
                break

        orchestrator_result = OrchestratorResult(
            conversation_id=conversation_id,
            objective=objective,
            status=status,
            objective_score=objective_score,
        )

        return orchestrator_result, prompt_request_response


    async def execute_multiple_steps_async(
            self,
            *,
            objectives: list[str],
            conversation_ids: Optional[list[str]] = None,
            expected_outputs: Optional[list[str]] = None,
            seed_prompts: Optional[list[SeedPromptGroup]] = None,
            prepended_conversations: Optional[list[list[PromptRequestResponse]]] = None,
            memory_labels: Optional[dict[str, str]] = None,
    ) -> list[OrchestratorResult]:
        """
        Runs multiple attacks in parallel using batch_size.
        Returns list of OrchestratorResult.
        """

        if not expected_outputs:
            expected_outputs = [None] * len(objectives)
        elif len(expected_outputs) != len(objectives):
            raise ValueError("Number of expected outputs must match number of objectives")

        if not seed_prompts:
            seed_prompts = [None] * len(objectives)
        elif len(seed_prompts) != len(objectives):
            raise ValueError("Number of seed prompts must match number of objectives")

        if not prepended_conversations:
            prepended_conversations = [None] * len(objectives)
        elif len(prepended_conversations) != len(objectives):
            raise ValueError("Number of prepended conversations must match number of objectives")

        if not conversation_ids:
            conversation_ids = [None] * len(objectives)
        elif len(conversation_ids) != len(objectives):
            raise ValueError("Number of conversation IDs must match number of objectives")

        batch_items: list[Sequence[Any]] = [
            objectives, expected_outputs, seed_prompts, prepended_conversations, conversation_ids
        ]
        batch_item_keys = [
            "objective", "expected_output", "seed_prompt", "prepended_conversation", "conversation_id"
        ]

        # Note: run_attack_async now returns (OrchestratorResult, PromptRequestResponse)
        results = await batch_task_async(
            prompt_target=self._objective_target,
            batch_size=self._batch_size,
            items_to_batch=batch_items,
            task_func=self.execute_step_async,
            task_arguments=batch_item_keys,
            memory_labels=memory_labels,
        )

        # Unpack the first item in the tuple, i.e., OrchestratorResult
        return [res[0] for res in results if res is not None and res[0] is not None]


    async def send_qa_pairs_async(self, qa_pairs: List[Dict[str, Any]]) -> None:
        """
        Sends a list of QA pairs using run_attack_async and run_attacks_async.
        Preserves thread management for multi-turn conversations.
        Returns a list of PromptRequestResponse objects.
        """
        single_turn_objectives = []
        single_turn_expected_outputs = []

        start_request_copy = self._objective_target.http_request

        for i, qa in enumerate(qa_pairs):
            print(f"\nExecuting test case: {i + 1}")
            self._objective_target.http_request = start_request_copy

            if "conversation" in qa:
                conversation_id = str(uuid.uuid4())
                is_thread_id_set = False

                for idx, turn in enumerate(qa["conversation"]):
                    prompt_text = turn["question"]
                    expected_output = turn["expected_outcome"]
                    print("Question:", prompt_text)

                    result, prompt_response = await self.execute_step_async(
                        objective=prompt_text,
                        expected_output=expected_output,
                        conversation_id=conversation_id
                    )

                    if not result:
                        continue

                    # Thread ID handling
                    if idx == 0 and not is_thread_id_set:
                        assistant_piece = next((p for p in prompt_response.request_pieces if p.role == "assistant"), None)
                        if assistant_piece and assistant_piece.prompt_metadata:
                            thread_id = assistant_piece.prompt_metadata.get("thread_id")
                            if thread_id:
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

                    await asyncio.sleep(1)

            else:
                # Single-turn QA
                single_turn_objectives.append(qa["question"])
                single_turn_expected_outputs.append(qa["expected_outcome"])

        # Run batched single-turn prompts
        if single_turn_objectives:
            await self.execute_multiple_steps_async(
                objectives=single_turn_objectives,
                expected_outputs=single_turn_expected_outputs,
            )


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

