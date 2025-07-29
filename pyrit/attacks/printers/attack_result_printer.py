# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from abc import ABC, abstractmethod

from pyrit.models import AttackResult


class AttackResultPrinter(ABC):
    """
    Abstract base class for printing attack results.

    This interface defines the contract for printing attack results in various formats.
    Implementations can render results to console, logs, files, or other outputs.
    """

    @abstractmethod
    async def print_result_async(self, result: AttackResult, *, include_auxiliary_scores: bool = False) -> None:
        """
        Print the complete attack result.

        Args:
            result (AttackResult): The attack result to print
            include_auxiliary_scores (bool): Whether to include auxiliary scores in the output.
                Defaults to False.
        """
        pass

    @abstractmethod
    async def print_conversation_async(self, result: AttackResult, *, include_auxiliary_scores: bool = False) -> None:
        """
        Print only the conversation history.

        Args:
            result (AttackResult): The attack result containing the conversation to print
            include_auxiliary_scores (bool): Whether to include auxiliary scores in the output.
                Defaults to False.
        """
        pass

    @abstractmethod
    async def print_summary_async(self, result: AttackResult) -> None:
        """
        Print a summary of the attack result without the full conversation.

        Args:
            result (AttackResult): The attack result to summarize
        """
        pass
