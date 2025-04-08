from __future__ import annotations
from abc import abstractmethod
import copy
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, TypeVar, Union
from pyrit.common import logger
from pyrit.memory.central_memory import CentralMemory
from pyrit.memory.memory_interface import MemoryInterface
from pyrit.models.seed_prompt import SeedPrompt
from pyrit.orchestratorv3.models.builder import AttackBuilder, OrchestratorBuilder
from pyrit.orchestratorv3.models.context import BacktrackingStrategy, MultiTurnAttackContext, MultiTurnAttackResult
from pyrit.prompt_converter.prompt_converter import PromptConverter
from pyrit.prompt_normalizer.prompt_normalizer import PromptNormalizer
from pyrit.prompt_target.common.prompt_chat_target import PromptChatTarget
from pyrit.prompt_target.common.prompt_target import PromptTarget
from pyrit.score.scorer import Scorer

T_AttBuilder = TypeVar('T_AttBuilder', bound='MultiTurnBaseAttackBuilder')
T_OrcBuilder = TypeVar('T_OrcBuilder', bound='MultiTurnBaseOrchestratorBuilder')

class MultiTurnBaseOrchestratorBuilder(OrchestratorBuilder[MultiTurnAttackContext, MultiTurnAttackResult]):
    """Base builder for multi-turn orchestrators with common configuration options."""
    
    def __init__(
        self,
        *,
        target: PromptTarget, 
        adversarial_chat: PromptChatTarget,
        system_prompt_path: Path,
        seed_prompt: Optional[str] = "How can I help you?",
        prompt_normalizer: Optional[PromptNormalizer] = None,
        memory: Optional[MemoryInterface] = None
    ):
        """Initialize the Multi-Turn Orchestrator Builder.
        
        Args:
            target: Target system to attack
            adversarial_chat: Adversarial chat target
            system_prompt_path: Path to the system prompt file
            seed_prompt: Initial prompt for the first turn
            prompt_normalizer: Optional prompt normalizer
            memory: Optional memory interface for storing attack data
        """
        self._context = MultiTurnAttackContext(
            target=target,
            adversarial_chat=adversarial_chat
        )
        self.with_seed_prompt(seed_prompt=seed_prompt)
        self.with_system_prompt(system_prompt=system_prompt_path)
        self._prompt_normalizer = prompt_normalizer or PromptNormalizer()
        self._memory = memory or CentralMemory.get_memory_instance()
    
    @property
    def context(self) -> MultiTurnAttackContext:
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
    
    def with_prompt_normalizer(self: T_OrcBuilder, prompt_normalizer: PromptNormalizer) -> T_OrcBuilder:
        """Set a prompt normalizer.
        
        Args:
            prompt_normalizer: Prompt normalizer instance
            
        Returns:
            Self for method chaining
        """
        if prompt_normalizer:
            self._prompt_normalizer = prompt_normalizer
        return self
    
    def with_max_turns(self: T_OrcBuilder, max_turns: int) -> T_OrcBuilder:
        """Set maximum conversation turns.
        
        Args:
            max_turns: Maximum number of conversation turns
            
        Returns:
            Self for method chaining
        """
        if max_turns < 1:
            raise ValueError("Maximum turns must be at least 1")
        
        self._context.max_turns = max_turns
        return self
    
    def with_backtracking(self: T_OrcBuilder, strategy: str) -> T_OrcBuilder:
        """Configure backtracking strategy.
        
        Args:
            strategy: Backtracking strategy name
            
        Returns:
            Self for method chaining
        """
        try:
            backtracking_strategy = BacktrackingStrategy(strategy)
            logger.info(f"Setting backtracking strategy to {backtracking_strategy}")
            self._context.backtracking_strategy = backtracking_strategy
        except ValueError:
            valid_strategies = ", ".join([s.value for s in BacktrackingStrategy])
            raise ValueError(
                f"Invalid backtracking strategy '{strategy}'. "
                f"Valid strategies are: {valid_strategies}"
            )
            
        return self
    
    def with_labels(self: T_OrcBuilder, labels: Dict[str, str]) -> T_OrcBuilder:
        """Set memory labels for the attack.
        
        Args:
            labels: Dictionary of label key-value pairs
            
        Returns:
            Self for method chaining
        """
        self._context.memory_labels = labels or []
        return self
    
    def with_custom_prompt(self: T_OrcBuilder, prompt: str) -> T_OrcBuilder:
        """Set a custom prompt for the first turn.
        
        Args:
            prompt: Custom prompt to use for first turn
            
        Returns:
            Self for method chaining
        """
        logger.info("Setting custom initial prompt")
        self._context.custom_prompt = prompt
        return self
    
    def with_system_prompt(self: T_OrcBuilder, system_prompt: Union[SeedPrompt | Path] ) -> T_OrcBuilder:
        """Set system prompt for the adversarial chat.
        
        Args:
            system_prompt: System prompt to use
            
        Returns:
            Self for method chaining
        """
        if isinstance(system_prompt, Path):
            sp = SeedPrompt.from_yaml_file(system_prompt)

            # if there is no 'objective' in the system prompt, raise an error
            if "objective" not in sp.parameters:
                raise ValueError(f"Adversarial seed prompt must have an objective: '{sp}'")
            self._system_prompt = sp
            
        elif isinstance(system_prompt, SeedPrompt):
            self._system_prompt = system_prompt

        else:
            raise ValueError("System prompt must be a SeedPrompt or Path")
        
        return self
    
    def with_seed_prompt(self: T_OrcBuilder, seed_prompt: Union[str | SeedPrompt]) -> T_OrcBuilder:
        """Set seed prompt for the adversarial chat.
        
        Args:
            seed_prompt: Seed prompt to use
            
        Returns:
            Self for method chaining
        """
        self._chat_seed_prompt = seed_prompt
        if isinstance(seed_prompt, str):
            self._chat_seed_prompt = SeedPrompt(value=seed_prompt, data_type="text")
        return self
    
    @abstractmethod
    def attack(self) -> AttackBuilder[MultiTurnAttackResult]:
        """Transition to attack configuration.
        
        Returns:
            An attack builder for configuring the attack
        """
        pass

class MultiTurnBaseAttackBuilder(AttackBuilder[MultiTurnAttackResult]):
    """Base builder for configuring multi-turn attacks with common configuration options."""
    
    def __init__(
        self, 
        *,
        context: MultiTurnAttackContext, 
        chat_system_seed_prompt: SeedPrompt,
        seed_prompt: SeedPrompt,
        prompt_normalizer: Optional[PromptNormalizer] = None,
        prepend_conversation: Optional[List[SeedPrompt]] = None,
        memory: Optional[MemoryInterface] = None,
    ):
        """Initialize the Red Teaming Attack Builder.
        
        Args:
            context: Context for the attack
            chat_system_seed_prompt: System prompt for the adversarial chat
            seed_prompt: Initial prompt for the first turn
            prompt_normalizer: Optional prompt normalizer for the attack
            prepend_conversation: Optional conversation to prepend
            memory: Optional memory interface for storing attack data
        """
        self._context = context
        self._chat_system_seed_prompt = chat_system_seed_prompt
        self._chat_seed_prompt = seed_prompt
        self._system_prompt = None
        self._prompt_normalizer = prompt_normalizer or PromptNormalizer()
        self._prepend_conversation = prepend_conversation or []
        self._memory = memory or CentralMemory.get_memory_instance()

    def with_prepended_conversation(self: T_AttBuilder, conversation: List[SeedPrompt]) -> T_AttBuilder:
        """Set a conversation to prepend to the attack.
        
        Args:
            conversation: List of SeedPrompt objects to prepend
            
        Returns:
            Self for method chaining
        """
        self._prepend_conversation = conversation or []
        return self
    
    def with_objective_scorer(self: T_AttBuilder, scorer: Scorer) -> T_AttBuilder:
        """Set the objective scorer for the attack.
        
        Args:
            scorer: Scorer to use for evaluating the attack
            
        Returns:
            Self for method chaining
        """
        self._objective_scorer = scorer
        return self

    def with_objective(self: T_AttBuilder, objective_text: str) -> T_AttBuilder:
        """Set the attack objective.
        
        Args:
            objective_text: Description of what the attack should achieve
            
        Returns:
            Self for method chaining
        """
        self._context.objective = objective_text
        return self
    
    def with_custom_prompt(self: T_AttBuilder, prompt: str) -> T_AttBuilder:
        """Set a custom prompt for the first turn.
        
        Args:
            prompt: Custom prompt to use for first turn
            
        Returns:
            Self for method chaining
        """
        self._context.custom_prompt = prompt
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
    def context(self) -> MultiTurnAttackContext:
        """Get the attack context being configured."""
        return copy.copy(self._context)
    
    @abstractmethod
    async def execute(self) -> MultiTurnAttackResult:
        """Execute the attack with the current configuration.
        
        Returns:
            Result of the attack
            
        Raises:
            ValueError: If the attack is not properly configured
        """
        pass

