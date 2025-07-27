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
    async def print_result(self, result: AttackResult) -> None:
        """
        Print the complete attack result.
        
        Args:
            result (AttackResult): The attack result to print
        """
        pass
    
    @abstractmethod
    async def print_conversation(self, result: AttackResult) -> None:
        """
        Print only the conversation history.
        
        Args:
            result (AttackResult): The attack result containing the conversation to print
        """
        pass
    
    @abstractmethod
    async def print_summary(self, result: AttackResult) -> None:
        """
        Print a summary of the attack result without the full conversation.
        
        Args:
            result (AttackResult): The attack result to summarize
        """
        pass