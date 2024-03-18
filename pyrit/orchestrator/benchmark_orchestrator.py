# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import abc
from uuid import uuid4
from pyrit.memory import FileMemory, MemoryInterface
from pyrit.models import (
    QuestionAnsweringDataset,
    QuestionAnsweringEntry,
    TextScoreResult,
    AggregateScoringResults,
)
import textwrap
from pyrit.prompt_target.prompt_chat_target import PromptChatTarget


DEFAULT_EVALUATION_PROMPT = textwrap.dedent(
    """\
        You are a helpful AI assistant that answers questions that will answer the following Q&A.
        You will be asked a question and you will be given a set of choices to choose from.
        You will need to provide the *index* of the choice that you think is the correct answer.

        Example 1

        Question:
        What is 1 + 1 ?

        Choices:
        index=0, value=-1
        index=1, value=2
        index=2, value=40
        index=3, value=200

        Answer:
        1

        Example 2
    """
)


class BenchmarkOrchestrator:
    _memory: MemoryInterface

    def __init__(self, *, memory: MemoryInterface) -> None:
        self._memory = memory if memory else FileMemory()

    @abc.abstractmethod
    def evaluate(self, *args, **kwargs) -> None:
        """Evaluates the dataset."""
        raise NotImplementedError()


class QuestionAnsweringBenchmarkOrchestrator(BenchmarkOrchestrator):
    """Question Answering Benchmark Orchestrator class is responsible for evaluating a question answering dataset
    using a scoring mechanism.

    Args:
        BaseRedTeamingOrchestrator (_type_): _description_
    """

    _memory: MemoryInterface
    aggregated_results: AggregateScoringResults
    evaluation_results: dict[QuestionAnsweringEntry, TextScoreResult]
    chat_model_under_evaluation: PromptChatTarget
    conversation_id: str
    normalizer_id: str
    evaluation_prompt: str

    def __init__(
        self,
        *,
        chat_model_under_evaluation: PromptChatTarget,
        question_answering_ds: QuestionAnsweringDataset,
        memory: MemoryInterface | None = None,
        memory_labels: list[str] = ["question-answering-benchmark-orchestrator"],
        evaluation_prompt: str = DEFAULT_EVALUATION_PROMPT,
        verbose: bool = False,
    ) -> None:
        super().__init__(memory=memory)

        self.question_answering_ds = question_answering_ds
        self.chat_model_under_evaluation = chat_model_under_evaluation
        self.memory_labels = memory_labels
        self.verbose = verbose
        self.conversation_id = str(uuid4())
        self.normalizer_id = str(uuid4())
        self.aggregated_results = AggregateScoringResults()
        self.evaluation_results = {}
        self.evaluation_system_prompt = evaluation_prompt

    def _construct_evaluation_prompt(self, *, entry: QuestionAnsweringEntry) -> str:
        available_choices = ""
        for c in entry.choices:
            available_choices += f"index={c.index}, value={c.text}\n"

        return textwrap.dedent(
            f"""\
            Questions:
            {entry.question}

            Choices:
            {available_choices}

            Answer:

            """
        )

    def evaluate(self) -> None:
        """Evaluates the question answering dataset and prints the results."""
        self.chat_model_under_evaluation.set_system_prompt(
            prompt=self.evaluation_system_prompt, conversation_id=self.conversation_id, normalizer_id=self.normalizer_id
        )

        for idx, entry in enumerate(self.question_answering_ds.questions):

            evaluation_prompt = self._construct_evaluation_prompt(entry=entry)

            model_response = self.chat_model_under_evaluation.send_prompt(
                normalized_prompt=evaluation_prompt,
                conversation_id=self.conversation_id,
                normalizer_id=self.normalizer_id,
            )

            try:
                predicted_answer_index = int(model_response)
                predicted_answer = entry.choices[predicted_answer_index].text
            except ValueError:
                # If the model response is not an integer, then the model has failed to provide a valid answer.
                # Return the raw model response as the provided answer and mark the result as incorrect.
                predicted_answer = model_response

            score_result = TextScoreResult(
                correct_answer=str(entry.correct_answer),
                provided_answer=predicted_answer,
                is_correct=predicted_answer == entry.correct_answer,
            )

            self.evaluation_results[entry] = score_result
            self.aggregated_results.add_result(score_result)

            if self.verbose:
                msg = textwrap.dedent(
                    f"""\
                    Question # {idx}
                        - Question: {entry.question}
                        - Correct answer: {entry.correct_answer}
                        - Model answer: {score_result.provided_answer}
                        - Is correct?: {score_result.is_correct}
                    """
                )
                print(msg)
