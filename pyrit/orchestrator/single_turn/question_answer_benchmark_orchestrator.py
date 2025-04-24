# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import pathlib
import uuid
from typing import Optional, Union

from pyrit.common.path import DATASETS_PATH
from pyrit.common.question_answer_helpers import construct_evaluation_prompt
from pyrit.common.utils import combine_dict
from pyrit.models import (
    PromptDataType,
    PromptRequestPiece,
    PromptRequestResponse,
    QuestionAnsweringDataset,
    SeedPromptDataset,
)
from pyrit.orchestrator import PromptSendingOrchestrator
from pyrit.prompt_converter import PromptConverter
from pyrit.prompt_normalizer import NormalizerRequest
from pyrit.prompt_target import PromptChatTarget
from pyrit.score import Scorer
from pyrit.score.question_answer_scorer import QuestionAnswerScorer


class QuestionAnsweringBenchmarkOrchestrator(PromptSendingOrchestrator):
    """
    Question Answering Benchmark Orchestrator class is responsible for evaluating a question answering dataset
    using a scoring mechanism.

    Args:
        Orchestrator (_type_): _description_
    """

    def __init__(
        self,
        objective_target: PromptChatTarget,
        question_answer_definition_path: pathlib.Path = pathlib.Path(DATASETS_PATH)
        / "orchestrators"
        / "benchmark"
        / "one_plus_one.yaml",
        scorers: Optional[list[Scorer]] = None,
        prompt_converters: Optional[list[PromptConverter]] = None,
        verbose: bool = False,
        batch_size: int = 10,
    ) -> None:
        """
        Initializes a QuestionAnsweringBenchmarkOrchestrator object.

        Args:
            objective_target (PromptChatTarget): The chat model to be evaluated.
            scorer (QuestionAnswerScorer): The scorer used to evaluate the chat model's responses.
            prompt_converters (list[PromptConverter], Optional): The prompt converters to be used.
            evaluation_prompt (str, Optional): The evaluation prompt to be used. Defaults to None.
            verbose (bool, Optional): Whether to print verbose output. Defaults to False.
        """

        question_answer_definition: SeedPromptDataset = SeedPromptDataset.from_yaml_file(
            question_answer_definition_path
        )

        self._user_start_turn = question_answer_definition.prompts[0]
        self._assistant_start_turn = question_answer_definition.prompts[1]

        super().__init__(
            objective_target=objective_target,
            scorers=scorers,
            prompt_converters=prompt_converters,
            verbose=verbose,
            batch_size=batch_size,
        )

        self._set_default_conversation_start()

    async def send_prompts_async(  # type: ignore[override]
        self,
        *,
        dataset: QuestionAnsweringDataset,
        prompt_type: PromptDataType = "text",
        memory_labels: Optional[dict[str, str]] = None,
        metadata: Optional[dict[str, Union[str, int]]] = None,
    ) -> list[PromptRequestResponse]:
        """Sends prompts to the chat model and evaluates responses."""
        prompt_list = [construct_evaluation_prompt(entry) for entry in dataset.questions]
        requests: list[NormalizerRequest] = []
        for prompt in prompt_list:

            requests.append(
                self._create_normalizer_request(
                    prompt_text=prompt,
                    prompt_type=prompt_type,
                    converters=self._prompt_converters,
                    metadata=metadata,
                    conversation_id=str(uuid.uuid4()),
                )
            )

        return await self.send_normalizer_requests_async(
            prompt_request_list=requests, memory_labels=memory_labels, dataset=dataset
        )

    async def send_normalizer_requests_async(  # type: ignore[override]
        self,
        *,
        prompt_request_list: list[NormalizerRequest],
        memory_labels: Optional[dict[str, str]] = None,
        dataset: QuestionAnsweringDataset,
    ) -> list[PromptRequestResponse]:
        """
        Sends the normalized prompts to the prompt target.
        """

        self.validate_normalizer_requests(prompt_request_list=prompt_request_list)

        for prompt in prompt_request_list:
            prompt.conversation_id = await self._prepare_conversation_async(normalizer_request=prompt)

        # Normalizer is responsible for storing the requests in memory
        # The labels parameter may allow me to stash class information for each kind of prompt.
        responses: list[PromptRequestResponse] = await self._prompt_normalizer.send_prompt_batch_to_target_async(
            requests=prompt_request_list,
            target=self._objective_target,
            labels=combine_dict(existing_dict=self._global_memory_labels, new_dict=memory_labels),
            orchestrator_identifier=self.get_identifier(),
            batch_size=self._batch_size,
        )

        if self._scorers and responses:
            response_pieces = PromptRequestResponse.flatten_to_prompt_request_pieces(responses)

            for scorer in self._scorers:
                if isinstance(scorer, QuestionAnswerScorer):
                    await scorer.score_prompts_with_tasks_batch_async(
                        request_responses=response_pieces, batch_size=self._batch_size, tasks=dataset.questions
                    )
                else:
                    await scorer.score_responses_inferring_tasks_batch_async(
                        request_responses=response_pieces, batch_size=self._batch_size
                    )

        return responses

    def _set_default_conversation_start(self):
        """Sets the default conversation start for the orchestrator."""
        prepended_conversation = [
            PromptRequestResponse(
                request_pieces=[
                    PromptRequestPiece(
                        role="user",
                        original_value=self._user_start_turn.value,
                    )
                ]
            ),
            PromptRequestResponse(
                request_pieces=[
                    PromptRequestPiece(
                        role="assistant",
                        original_value=self._assistant_start_turn.value,
                    )
                ]
            ),
        ]

        self.set_prepended_conversation(prepended_conversation=prepended_conversation)
