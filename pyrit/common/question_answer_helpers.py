# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from pyrit.models import QuestionAnsweringEntry


def construct_evaluation_prompt(entry: QuestionAnsweringEntry) -> str:
    """
    From question and choices in entry, creates prompt to be send to target

    Args:
        entry (QuestionAnsweringEntry): A single entry from which the prompt is constructed
    """
    available_choices = "\n".join(f"index={c.index}, value={c.text}" for c in entry.choices)
    prompt = f"Question:\n" f"{entry.question}\n\n" f"Choices:\n" f"{available_choices}"
    return prompt
