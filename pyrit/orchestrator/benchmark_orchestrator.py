# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import textwrap
import yaml
from uuid import uuid4
from pyrit.memory import MemoryInterface
from pyrit.score.question_answer_scorer import QuestionAnswerScorer
from pyrit.prompt_target import PromptChatTarget
from pyrit.orchestrator.prompt_sending_orchestrator import PromptSendingOrchestrator
from pyrit.common.path import DATASETS_PATH
from pathlib import Path


class QuestionAnsweringBenchmarkOrchestrator(PromptSendingOrchestrator):
    """Question Answering Benchmark Orchestrator class is responsible for evaluating a question answering dataset
    using a scoring mechanism.

    Args:
        BaseRedTeamingOrchestrator (_type_): _description_
    """

    _memory: MemoryInterface
    chat_model_under_evaluation: PromptChatTarget
    conversation_id: str
    normalizer_id: str
    evaluation_prompt: str

    def __init__(
        self,
        *,
        chat_model_under_evaluation: PromptChatTarget,
        scorer: QuestionAnswerScorer,
        memory: MemoryInterface | None = None,
        memory_labels: list[str] = ["question-answering-benchmark-orchestrator"],
        evaluation_prompt: str | None = None,
        batch_size: int = 1,
        verbose: bool = False,
        include_original_prompts: bool = False,
    ) -> None:
        """
        Initializes a BenchmarkOrchestrator object.

        Args:
            chat_model_under_evaluation (PromptChatTarget): The chat model to be evaluated.
            scorer (QuestionAnswerScorer): The scorer used to evaluate the chat model's responses.
            memory (MemoryInterface | None, optional): The memory interface to be used. Defaults to None.
            memory_labels (list[str], optional): The labels to be associated with the memory.
            Defaults to ["question-answering-benchmark-orchestrator"].
            evaluation_prompt (str | None, optional): The evaluation prompt to be used. Defaults to None.
            batch_size (int, optional): The batch size for evaluation. Defaults to 1.
            verbose (bool, optional): Whether to print verbose output. Defaults to False.
            include_original_prompts (bool, optional): Whether to include original prompts in the evaluation.
            Defaults to False.
        """
        self.chat_model_under_evaluation = chat_model_under_evaluation
        self.scorer = scorer
        self.memory_labels = memory_labels
        self.conversation_id = str(uuid4())
        self.normalizer_id = str(uuid4())

        if evaluation_prompt:
            self.evaluation_system_prompt = evaluation_prompt
        else:
            default_data_path = Path(DATASETS_PATH, "orchestrators", "question_answer_default.yaml")
            default_data = default_data_path.read_text(encoding="utf-8")
            yamp_data = yaml.safe_load(default_data)
            self.evaluation_system_prompt = yamp_data.get("content")

        super().__init__(
            prompt_target=chat_model_under_evaluation,
            batch_size=batch_size,
            prompt_converters=None,
            memory=memory,
            include_original_prompts=include_original_prompts,
            verbose=verbose,
        )

    def evaluate(self) -> None:
        """Evaluates the question answering dataset and prints the results."""
        self.chat_model_under_evaluation.set_system_prompt(
            prompt=self.evaluation_system_prompt,
            conversation_id=self.conversation_id,
            normalizer_id=self.normalizer_id,
        )

        for idx, (question_entry, question_prompt) in enumerate(self.scorer.get_next_question_prompt_pair()):
            # NOTE. We are not using `send_prompt` method here, because unsure on how to set the system prompt
            # If we use this without the system prompt, the model doesn't not perform as well as expected because
            # it doesn't have the instructions for how it should respond to the questions.
            # model_responses = self.send_prompts([question_prompt])
            model_response = self.chat_model_under_evaluation.send_prompt(
                normalized_prompt=question_prompt,
                conversation_id=self.conversation_id,
                normalizer_id=self.normalizer_id,
            )

            curr_score = self.scorer.score_question(question=question_entry, answer=model_response)

            if self._verbose:
                msg = textwrap.dedent(
                    f"""\
                    Question # {idx}
                        - Question: {question_entry}
                        - Score: {curr_score}
                    """
                )
                print(msg)
