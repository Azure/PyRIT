from __future__ import annotations
from abc import abstractmethod
import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, TypeVar, Union
import uuid

from pyrit.memory.central_memory import CentralMemory
from pyrit.memory.memory_interface import MemoryInterface
from pyrit.models.filter_criteria import PromptConverterState
from pyrit.models.literals import PromptDataType
from pyrit.models.prompt_request_response import PromptRequestResponse
from pyrit.models.seed_prompt import SeedPrompt, SeedPromptGroup
from pyrit.orchestratorv3.models.builder import AttackBuilder, OrchestratorBuilder
from pyrit.orchestratorv3.models.context import SingleTurnAttackContext, SingleTurnAttackResult
from pyrit.prompt_converter.prompt_converter import PromptConverter
from pyrit.prompt_normalizer.normalizer_request import NormalizerRequest
from pyrit.prompt_normalizer.prompt_converter_configuration import PromptConverterConfiguration
from pyrit.prompt_normalizer.prompt_normalizer import PromptNormalizer
from pyrit.prompt_target.common.prompt_target import PromptTarget
from pyrit.score.scorer import Scorer

T_AttBuilder = TypeVar('T_AttBuilder', bound='SingleTurnBaseAttackBuilder')
T_OrcBuilder = TypeVar('T_OrcBuilder', bound='SingleTurnBaseOrchestratorBuilder')

class SingleTurnBaseOrchestratorBuilder(OrchestratorBuilder[SingleTurnAttackContext, SingleTurnAttackResult]):
    """Base builder for single-turn orchestrators with common configuration options."""
    
    def __init__(
        self,
        *,
        target: PromptTarget, 
        batch_size: int = 10,
        prompt_normalizer: Optional[PromptNormalizer] = None,
        memory: Optional[MemoryInterface] = None
    ):
        """Initialize the Single-Turn Orchestrator Builder.
        
        Args:
            target: Target system to attack
            batch_size: Maximum batch size for sending prompts
            prompt_normalizer: Optional prompt normalizer
            memory: Optional memory interface for storing attack data
        """
        self._context = SingleTurnAttackContext(
            target=target,
            batch_size=batch_size
        )
        self._prompt_normalizer = prompt_normalizer or PromptNormalizer()
        self._memory = memory or CentralMemory.get_memory_instance()
    
    @property
    def context(self) -> SingleTurnAttackContext:
        """Get the attack context being built."""
        return copy.copy(self._context)
    
    def with_prompt_converters(self: T_OrcBuilder, prompt_converters: List[PromptConverter]) -> T_OrcBuilder:
        """Set custom prompt converters.
        
        Args:
            prompt_converters: List of prompt converter instances
            
        Returns:
            Self for method chaining
        """
        self._context.prompt_converters = prompt_converters or []
        return self
    
    def with_batch_size(self: T_OrcBuilder, batch_size: int) -> T_OrcBuilder:
        """Set batch size for sending prompts.
        
        Args:
            batch_size: Maximum batch size
            
        Returns:
            Self for method chaining
        """
        if batch_size < 1:
            raise ValueError("Batch size must be at least 1")
        
        self._context.batch_size = batch_size
        return self
    
    def with_labels(self: T_OrcBuilder, labels: Dict[str, str]) -> T_OrcBuilder:
        """Set memory labels for the attack.
        
        Args:
            labels: Dictionary of label key-value pairs
            
        Returns:
            Self for method chaining
        """
        self._context.memory_labels = labels
        return self
    
    def with_prompts(self: T_OrcBuilder, prompts: List[str]) -> T_OrcBuilder:
        """Set the prompts to send.
        
        Args:
            prompts: List of prompts
            
        Returns:
            Self for method chaining
        """
        self._context.prompts = prompts or []
        return self
    
    def with_skip_criteria(
        self: T_OrcBuilder, 
        *,
        skip_criteria: PromptConverter,
        skip_value_type: PromptConverterState = "original"
    ) -> T_OrcBuilder:
        """Set the skip criteria for the orchestrator.
        
        Args:
            skip_criteria: Criteria to skip prompts
            skip_value_type: Type of value to skip
            
        Returns:
            Self for method chaining
        """
        self._prompt_normalizer.set_skip_criteria(skip_criteria=skip_criteria, skip_value_type=skip_value_type)
        return self
    
    def with_metadata(self: T_OrcBuilder, metadata: Dict[str, Union[str, int]]) -> T_OrcBuilder:
        """Set metadata for the attack.
        
        Args:
            metadata: Dictionary of metadata key-value pairs
            
        Returns:
            Self for method chaining
        """
        self._context.metadata = metadata or {}
        return self
    
    @abstractmethod
    def attack(self) -> AttackBuilder[SingleTurnAttackResult]:
        """Transition to attack configuration.
        
        Returns:
            An attack builder for configuring the attack
        """
        pass

class SingleTurnBaseAttackBuilder(AttackBuilder[SingleTurnAttackResult]):
    """Base builder for configuring single-turn attacks."""
    
    def __init__(
        self, 
        *,
        context: SingleTurnAttackContext,
        prompt_normalizer: Optional[PromptNormalizer] = None,
        memory: Optional[MemoryInterface] = None,
    ):
        """Initialize the Single Turn Attack Builder.
        
        Args:
            context: Context for the attack
            prompt_normalizer: Optional prompt normalizer for the attack
            memory: Optional memory interface for storing attack data
        """
        self._context = context
        self._prompt_normalizer = prompt_normalizer or PromptNormalizer()
        self._memory = memory or CentralMemory.get_memory_instance()
        self._scorers = []

    def with_scorers(self: T_AttBuilder, scorers: List[Scorer]) -> T_AttBuilder:
        """Set the scorers for the attack.
        
        Args:
            scorers: List of scorers to use for evaluating the attack
            
        Returns:
            Self for method chaining
        """
        self._scorers = scorers or []
        return self
    
    def with_prompts(self: T_AttBuilder, prompts: List[str]) -> T_AttBuilder:
        """Set the prompts to send.
        
        Args:
            prompts: List of prompts
            
        Returns:
            Self for method chaining
        """
        self._context.prompts = prompts or []
        return self
    
    def with_prepended_conversation(self, conversation: List[PromptRequestResponse]) -> T_AttBuilder:
        """Set a conversation to prepend to the attack.
        
        Args:
            conversation: List of PromptRequestResponse objects to prepend
            
        Returns:
            Self for method chaining
        """
        if conversation and not isinstance(self._context.target, PromptTarget):
            raise TypeError(
               f"Only PromptChatTargets are able to modify conversation history. Instead objective_target is: "
                f"{type(self._context.target)}."
            )
        self._prepended_conversation = conversation
        return self
    
    def with_labels(self: T_AttBuilder, labels: Dict[str, str]) -> T_AttBuilder:
        """Set memory labels for the attack.
        
        Args:
            labels: Dictionary of label key-value pairs
            
        Returns:
            Self for method chaining
        """
        self._context.memory_labels = labels
        return self
    
    @property
    def context(self) -> SingleTurnAttackContext:
        """Get the attack context being configured."""
        return copy.copy(self._context)
    
    @abstractmethod
    async def execute(self) -> SingleTurnAttackResult:
        """Execute the attack with the current configuration.
        
        Returns:
            Result of the attack
            
        Raises:
            ValueError: If the attack is not properly configured
        """
        pass