# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import asyncio
import logging

from tqdm import tqdm

from pyrit.interfaces import CompletionSupport, SupportTextClassification
from pyrit.models import ChatMessage, PromptDataset, PromptResponse, Score

log = logging.getLogger(__name__)


def score_text(
    messages: list[ChatMessage],
    classifier: SupportTextClassification,
    verbose: bool = True,
) -> list[Score]:
    """Score a list of messages using a classifier

    Args:
        messages: The messages to score
        classifier: The classifier to use

    Returns:
        A list of scores
    """
    scores: list[Score] = []
    for m in (pbar := tqdm(messages, disable=not verbose)):
        pbar.set_description("Scoring messages")
        score = classifier.score_text(m.content)
        scores.append(score)
    return scores


def evaluate(target: CompletionSupport, prompt_dataset: PromptDataset, verbose: bool = True) -> list[PromptResponse]:
    """Evaluate a target model on a prompt dataset

    Args:
        target: The target model to evaluate
        prompt_dataset: The prompt dataset to evaluate on
        memory: The memory to use for the evaluation (if any)

    Returns:

    """
    evaluation_chats: list[PromptResponse] = []

    for p in (pbar := tqdm(prompt_dataset.prompts, disable=not verbose)):
        pbar.set_description("Evaluating prompts")
        target_prompt_response = target.complete_text(p)
        evaluation_chats.append(target_prompt_response)
    return evaluation_chats


async def evaluate_async(
    target: CompletionSupport,
    prompt_dataset: PromptDataset,
    verbose: bool = True,
    max_concurrent: int = 10,
) -> list[PromptResponse]:
    """Evaluate a target model on a prompt dataset

    Args:
        target: The target model to evaluate
        prompt_dataset: The prompt dataset to evaluate on

    Returns:
        A list of PromptResponse after evaluating the model on the dataset.
    """

    async def process_prompt(prompt: str) -> PromptResponse:
        return await target.complete_text_async(prompt)

    evaluation_chats: list[PromptResponse] = []
    tasks = []
    exceptions: list[BaseException] = []

    for p in (pbar := tqdm(prompt_dataset.prompts, disable=not verbose)):
        pbar.set_description("Evaluating prompts")
        tasks.append(process_prompt(p))
        if len(tasks) >= max_concurrent:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            _separate_exceptions_from_successful_chats(results, evaluation_chats, exceptions)
            tasks.clear()
    if tasks:
        _separate_exceptions_from_successful_chats(results, evaluation_chats, exceptions)

    for exception in exceptions:
        if verbose:
            log.info(f"Error processing the prompt: {exception}")

    return evaluation_chats


def _separate_exceptions_from_successful_chats(
    results: list[PromptResponse | BaseException],
    chats: list[PromptResponse],
    exceptions: list[BaseException],
):
    """Split results into exceptions and successful chats."""
    for result in results:
        chats.append(result) if isinstance(result, PromptResponse) else exceptions.append(result)
