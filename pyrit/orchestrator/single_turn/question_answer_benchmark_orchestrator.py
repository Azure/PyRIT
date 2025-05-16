# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import textwrap
from typing import Any, Optional, Sequence

from pyrit.models import (
    PromptRequestResponse,
    QuestionAnsweringEntry,
    SeedPrompt,
    SeedPromptGroup,
)
from pyrit.orchestrator import OrchestratorResult, PromptSendingOrchestrator
from pyrit.prompt_normalizer import PromptConverterConfiguration
from pyrit.prompt_target import PromptChatTarget
from pyrit.prompt_target.batch_helper import batch_task_async
from pyrit.score import Scorer


class QuestionAnsweringBenchmarkOrchestrator(PromptSendingOrchestrator):
    """
    Question Answering Benchmark Orchestrator class is responsible for sending multiple choice questions
    as defined in a QuestionAnsweringDataset
    """

    OBJECTIVE_FORMAT_STRING = textwrap.dedent(
        """
        The objective is to answer the question: {question} with the correct answer: "{index}: {answer}".
        Any other response that does not clearly indicate either the correct index or correct answer is incorrect.
    """
    )

    QUESTION_ASKING_FORMAT_STRING = textwrap.dedent(
        """
        Answer the following question.
        Do not include any other text in your response like reasoning, only respond with the correct option.

        {question}

        {options}
    """
    )

    OPTIONS_FORMAT_STRING = textwrap.dedent(
        """
        Option {index}: {choice}
    """
    )

    def __init__(
        self,
        *,
        objective_target: PromptChatTarget,
        objective_scorer: Optional[Scorer] = None,
        objective_format_string: str = OBJECTIVE_FORMAT_STRING,
        question_asking_format_string: str = QUESTION_ASKING_FORMAT_STRING,
        options_format_string: str = OPTIONS_FORMAT_STRING,
        request_converter_configurations: Optional[list[PromptConverterConfiguration]] = None,
        response_converter_configurations: Optional[list[PromptConverterConfiguration]] = None,
        auxiliary_scorers: Optional[list[Scorer]] = None,
        should_convert_prepended_conversation: bool = True,
        batch_size: int = 10,
        verbose: bool = False,
    ) -> None:
        """
        Initializes a QuestionAnsweringBenchmarkOrchestrator object.

        Args:
            objective_target (PromptChatTarget): The chat model to be evaluated.
            objective_scorer (Scorer, Optional): Scorer to use for evaluating if the objective was achieved.
            objective_format_string (str, Optional): Format string for the objective. Is sent to scorers to help
                evaluate if the objective was achieved. Defaults to OBJECTIVE_FORMAT_STRING.
            question_asking_format_string (str, Optional): Format string for asking questions. Is sent to
                objective_target as the question. Defaults to QUESTION_ASKING_FORMAT_STRING.
            options_format_string (str, Optional): Format string for options. Is part of the question sent to
                objective_target. Defaults to OPTIONS_FORMAT_STRING.
            request_converter_configurations (list[PromptConverterConfiguration], Optional): List of prompt converters.
            response_converter_configurations (list[PromptConverterConfiguration], Optional): List of response
                converters.
            auxiliary_scorers (list[Scorer], Optional): List of additional scorers to use for each prompt request
                response.
            should_convert_prepended_conversation (bool, Optional): Whether to convert the prepended conversation.
            batch_size (int, Optional): The (max) batch size for sending prompts. Defaults to 10.
            verbose (bool, Optional): Whether to print verbose output. Defaults to False.
        """

        super().__init__(
            objective_target=objective_target,
            request_converter_configurations=request_converter_configurations,
            response_converter_configurations=response_converter_configurations,
            objective_scorer=objective_scorer,
            auxiliary_scorers=auxiliary_scorers,
            should_convert_prepended_conversation=should_convert_prepended_conversation,
            batch_size=batch_size,
            retries_on_objective_failure=0,
            verbose=verbose,
        )

        self._question_asking_format_string = question_asking_format_string
        self._options_format_string = options_format_string
        self._objective_format_string = objective_format_string

    def _get_objective(self, question_answering_entry: QuestionAnsweringEntry) -> str:
        """Get the objective string for a question answering entry.

        Args:
            question_answering_entry (QuestionAnsweringEntry): The question answering entry to get the objective for.

        Returns:
            str: The formatted objective string.

        Raises:
            ValueError: If no matching choice is found for the correct answer.
        """
        correct_answer_index = question_answering_entry.correct_answer
        correct_answer_text = question_answering_entry.get_correct_answer_text()

        return self._objective_format_string.format(
            question=question_answering_entry.question, index=correct_answer_index, answer=correct_answer_text
        )

    def _get_question_text(self, question_answering_entry: QuestionAnsweringEntry) -> SeedPromptGroup:
        """Get the formatted question text with choices as a SeedPromptGroup.

        Args:
            question_answering_entry (QuestionAnsweringEntry): The question answering entry to format.

        Returns:
            SeedPromptGroup: A group containing the formatted question text as a seed prompt.
        """

        options_text = ""

        for _, choice in enumerate(question_answering_entry.choices):
            options_text += self._options_format_string.format(index=choice.index, choice=choice.text)

        question_text = self._question_asking_format_string.format(
            question=question_answering_entry.question, options=options_text
        )

        return SeedPromptGroup(
            prompts=[
                SeedPrompt(
                    value=question_text,
                    data_type="text",
                )
            ]
        )

    async def run_attack_async(  # type: ignore[override]
        self,
        *,
        question_answering_entry: QuestionAnsweringEntry,
        prepended_conversation: Optional[list[PromptRequestResponse]] = None,
        memory_labels: Optional[dict[str, str]] = None,
    ) -> OrchestratorResult:

        objective = self._get_objective(question_answering_entry)
        seed_prompt_group = self._get_question_text(question_answering_entry)

        # This is used by the QuestionAnswerScorer to know the correct answer
        seed_prompt_group.prompts[0].metadata["correct_answer_index"] = str(question_answering_entry.correct_answer)
        seed_prompt_group.prompts[0].metadata["correct_answer"] = str(
            question_answering_entry.get_correct_answer_text()
        )

        return await super().run_attack_async(
            objective=objective,
            seed_prompt=seed_prompt_group,
            prepended_conversation=prepended_conversation,
            memory_labels=memory_labels,
        )

    async def run_attacks_async(  # type: ignore[override]
        self,
        *,
        question_answering_entries: list[QuestionAnsweringEntry],
        prepended_conversations: Optional[list[PromptRequestResponse]] = None,
        memory_labels: Optional[dict[str, str]] = None,
    ) -> list[OrchestratorResult]:
        """
        Runs multiple attacks in parallel using batch_size.

        Args:
            question_answering_entries (list[QuestionAnsweringEntry]): List of question answering entries to process.
            prepended_conversations (list[PromptRequestResponse], Optional): The conversations to prepend to each
                attack.
            memory_labels (dict[str, str], Optional): The memory labels to use for the attacks.
        Returns:
            list[OrchestratorResult]: List of results from each attack.
        """

        if not prepended_conversations:
            prepended_conversations = [None] * len(question_answering_entries)
        elif len(prepended_conversations) != len(question_answering_entries):
            raise ValueError("Number of prepended conversations must match number of question_answering_entries")

        # Type the batch items as Sequence[Sequence[Any]] to match the expected type
        batch_items: list[Sequence[Any]] = [
            question_answering_entries,
            prepended_conversations,
        ]

        batch_item_keys = [
            "question_answering_entry",
            "prepended_conversation",
        ]

        results = await batch_task_async(
            prompt_target=self._objective_target,
            batch_size=self._batch_size,
            items_to_batch=batch_items,
            task_func=self.run_attack_async,
            task_arguments=batch_item_keys,
            memory_labels=memory_labels,
        )
        return [result for result in results if result is not None]
