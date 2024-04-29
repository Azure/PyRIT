# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import textwrap
from typing import Optional
import yaml
from uuid import uuid4
from pyrit.memory import MemoryInterface
from pyrit.orchestrator.orchestrator_class import Orchestrator
from pyrit.prompt_converter import PromptConverter
from pyrit.prompt_normalizer import PromptNormalizer
from pyrit.score.question_answer_scorer import QuestionAnswerScorer
from pyrit.prompt_target import PromptChatTarget
from pyrit.common.path import DATASETS_PATH
from pathlib import Path


class QuestionAnsweringBenchmarkOrchestrator(Orchestrator):
    """Question Answering Benchmark Orchestrator class is responsible for evaluating a question answering dataset
    using a scoring mechanism.

    Args:
        BaseRedTeamingOrchestrator (_type_): _description_
    """

    _memory: MemoryInterface
    _chat_model_under_evaluation: PromptChatTarget
    _conversation_id: str
    normalizer_id: str
    evaluation_prompt: str

    def __init__(
        self,
        *,
        chat_model_under_evaluation: PromptChatTarget,
        scorer: QuestionAnswerScorer,
        prompt_converters: list[PromptConverter] = [],
        memory: Optional[MemoryInterface] = None,
        memory_labels: Optional[dict[str, str]] = None,
        evaluation_prompt: Optional[str] = None,
        verbose: bool = False,
    ) -> None:
        """
        Initializes a BenchmarkOrchestrator object.

        Args:
            chat_model_under_evaluation (PromptChatTarget): The chat model to be evaluated.
            scorer (QuestionAnswerScorer): The scorer used to evaluate the chat model's responses.
            prompt_converters (list[PromptConverter], optional): The prompt converters to be used.
            memory (MemoryInterface, optional): The memory interface to be used. Defaults to None.
            memory_labels (dict[str, str], optional): The labels to be associated with the memory.
                Defaults to ["question-answering-benchmark-orchestrator"].
            evaluation_prompt (str, optional): The evaluation prompt to be used. Defaults to None.
            verbose (bool, optional): Whether to print verbose output. Defaults to False.
        """
        super().__init__(
            prompt_converters=prompt_converters,
            memory=memory,
            verbose=verbose,
            memory_labels=memory_labels,
        )

        self._chat_model_under_evaluation = chat_model_under_evaluation
        self._scorer = scorer
        self._conversation_id = str(uuid4())
        self._normalizer = PromptNormalizer(memory=self._memory)

        if evaluation_prompt:
            self.evaluation_system_prompt = evaluation_prompt
        else:
            default_data_path = Path(DATASETS_PATH, "orchestrators", "benchmark", "question_answer.yaml")
            default_data = default_data_path.read_text(encoding="utf-8")
            yamp_data = yaml.safe_load(default_data)
            self.evaluation_system_prompt = yamp_data.get("content")

    def evaluate(self) -> None:
        """Evaluates the question answering dataset and prints the results."""
        self._chat_model_under_evaluation.set_system_prompt(
            system_prompt=self.evaluation_system_prompt,
            conversation_id=self._conversation_id,
            orchestrator_identifier=self.get_identifier(),
            labels=self._global_memory_labels,
        )

        for idx, (question_entry, question_prompt) in enumerate(self._scorer.get_next_question_prompt_pair()):

            request = self._create_normalizer_request(question_prompt, "text")

            response = self._normalizer.send_prompt(
                normalizer_request=request,
                target=self._chat_model_under_evaluation,
                conversation_id=self._conversation_id,
                labels=self._global_memory_labels,
                orchestrator_identifier=self.get_identifier(),
            )

            answer = response.request_pieces[0].converted_value
            curr_score = self._scorer.score_question(question=question_entry, answer=answer)

            if self._verbose:
                msg = textwrap.dedent(
                    f"""\
                    Question # {idx}
                        - Question: {question_entry}
                        - Score: {curr_score}
                    """
                )
                print(msg)
