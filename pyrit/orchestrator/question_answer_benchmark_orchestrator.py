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

class QuestionAnswerPaths(enum.Enum):
    DEFAULT = pathlib.Path(DATASETS_PATH) / "orchestrators" / "benchmark" / "question_answer.yaml"

class QuestionAnsweringBenchmarkOrchestrator(PromptSendingOrchestrator):
    """Question Answering Benchmark Orchestrator class is responsible for evaluating a question answering dataset
    using a scoring mechanism.

    Args:
        Orchestrator (_type_): _description_
    """

    def __init__(
        self,
        objective_target: PromptChatTarget,
        question_answer_definition_path: pathlib.Path = QuestionAnswerPaths.DEFAULT.value,
        scorers: Optional[list[Scorer]] = None,
        prompt_converters: Optional[list[PromptConverter]] = None,
        evaluation_prompt: Optional[str] = None,
        verbose: bool = False,
    ) -> None:
        """
        Initializes a QuestionAnsweringBenchmarkOrchestrator object.

        Args:
            chat_model_under_evaluation (PromptChatTarget): The chat model to be evaluated.
            scorer (QuestionAnswerScorer): The scorer used to evaluate the chat model's responses.
            prompt_converters (list[PromptConverter], Optional): The prompt converters to be used.
            evaluation_prompt (str, Optional): The evaluation prompt to be used. Defaults to None.
            verbose (bool, Optional): Whether to print verbose output. Defaults to False.
        """

        question_answer_definition: SeedPromptDataset = SeedPromptDataset.from_yaml_file(question_answer_definition_path)

        self._user_start_turn = question_answer_definition.prompts[0]
        self._assistant_start_turn = question_answer_definition.prompts[1]

        if evaluation_prompt:
            self.evaluation_system_prompt = evaluation_prompt
        else:
            default_data_path = Path(DATASETS_PATH, "orchestrators", "benchmark", "question_answer.yaml")
            default_data = default_data_path.read_text(encoding="utf-8")
            yamp_data = yaml.safe_load(default_data)
            self.evaluation_system_prompt = yamp_data.get("content")

        super().__init__(
            objective_target=objective_target,
            scorers=scorers,
            prompt_converters=prompt_converters,
            verbose=verbose,
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

    # async def evaluate(self) -> None:
    #     """Evaluates the question answering dataset and prints the results."""
    #     self._chat_model_under_evaluation.set_system_prompt(
    #         system_prompt=self.evaluation_system_prompt,
    #         conversation_id=self._conversation_id,
    #         orchestrator_identifier=self.get_identifier(),
    #         labels=self._global_memory_labels,
    #     )

    #     for idx, (question_entry, question_prompt) in enumerate(self._scorer.get_next_question_prompt_pair()):

    #         seed_prompt_group = SeedPromptGroup(
    #             prompts=[
    #                 SeedPrompt(
    #                     value=question_prompt,
    #                     data_type="text",
    #                 )
    #             ]
    #         )

    #         response = await self._normalizer.send_prompt_async(
    #             seed_prompt_group=seed_prompt_group,
    #             conversation_id=self._conversation_id,
    #             target=self._chat_model_under_evaluation,
    #             labels=self._global_memory_labels,
    #             orchestrator_identifier=self.get_identifier(),
    #         )

    #         answer = response.request_pieces[0].converted_value
    #         curr_score = self._scorer.score_question(question=question_entry, answer=answer)

    #         if self._verbose:
    #             msg = textwrap.dedent(
    #                 f"""\
    #                 Question # {idx}
    #                     - Question: {question_entry}
    #                     - Score: {curr_score}
    #                 """
    #             )
    #             print(msg)

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