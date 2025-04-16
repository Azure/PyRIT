# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import textwrap
from pathlib import Path
from typing import Optional, List, Tuple
import pathlib
import enum

import yaml

from pyrit.common.path import DATASETS_PATH
from pyrit.models import SeedPrompt, PromptRequestPiece, PromptRequestResponse, SeedPromptGroup, SeedPromptDataset, QuestionAnsweringDataset, QuestionAnsweringEntry
from pyrit.orchestrator import PromptSendingOrchestrator
from pyrit.prompt_converter import PromptConverter
from pyrit.prompt_target import PromptChatTarget
from pyrit.score import Scorer
from pyrit.score.question_answer_scorer import QuestionAnswerScorer
from pyrit.prompt_normalizer import NormalizerRequest
from pyrit.common.question_answer_helpers import get_question_prompt_pairs

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
        question_answer_definition_path: pathlib.Path = pathlib.Path(DATASETS_PATH) / "orchestrators" / "benchmark" / "one_plus_one.yaml",
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

        question_answer_definition: SeedPromptDataset = SeedPromptDataset.from_yaml_file(question_answer_definition_path)

        self._user_start_turn = question_answer_definition.prompts[0]
        self._assistant_start_turn = question_answer_definition.prompts[1]

        super().__init__(
            objective_target=objective_target,
            scorers=scorers,
            prompt_converters=prompt_converters,
            verbose=verbose,
            batch_size=batch_size
        )

        self._set_default_conversation_start()

    async def send_prompts_async(
        self, 
        dataset: QuestionAnsweringDataset
    ) -> list[PromptRequestResponse]:
        """Sends prompts to the chat model and evaluates responses."""
        qa_request_list = get_question_prompt_pairs(dataset=dataset)
        prompt_list = [entry[1] for entry in qa_request_list]
        return await super().send_prompts_async(prompt_list=prompt_list, prompt_type="text")

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