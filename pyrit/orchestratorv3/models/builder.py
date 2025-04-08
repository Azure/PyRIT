from __future__ import annotations
from abc import ABC, abstractmethod
import ast
from dataclasses import dataclass, field
from typing import Generic, List, Mapping, Optional, TypeVar
import uuid
from pyrit.common import default_values
from pyrit.memory.central_memory import CentralMemory
from pyrit.memory.memory_interface import MemoryInterface
from pyrit.models.seed_prompt import SeedPrompt
from pyrit.orchestratorv3.models.context import AttackContext, AttackResult
from pyrit.prompt_normalizer.prompt_normalizer import PromptNormalizer
from pyrit.prompt_target.common.prompt_chat_target import PromptChatTarget
from pyrit.prompt_target.common.prompt_target import PromptTarget

# Type variables with bounds for proper static typing
T_Context = TypeVar('T_Context', bound=AttackContext)
T_Result = TypeVar('T_Result', bound=AttackResult)

class OrchestratorBuilderFactory(ABC, Generic[T_Context, T_Result]):
    """Factory for creating orchestrator builders."""
    
    @classmethod
    @abstractmethod
    def create(cls, **kwargs) -> OrchestratorBuilder[T_Context, T_Result]:
        """
        Create a new orchestrator builder.
        
        Returns:
            A configured builder
        """
        pass

class OrchestratorBuilder(ABC, Generic[T_Context, T_Result]):
    """Base builder for orchestrator configuration."""
    
    # This should maintain the context being built
    @property
    @abstractmethod
    def context(self) -> T_Context:
        """Get the context being built."""
        pass

    @abstractmethod
    def attack(self) -> 'AttackBuilder[T_Result]':
        """
        Transition to attack configuration.
        
        Returns:
            An attack builder that works with this builder's context
        """
        pass

class AttackBuilder(ABC, Generic[T_Result]):
    """Base builder for attack configuration."""
    
    @abstractmethod
    async def execute(self) -> T_Result:
        """
        Execute the attack with the current configuration.
        
        Returns:
            Attack result of type T_Result
        """
        pass

@dataclass(frozen=True)
class AttackStrategyIdentifier(Mapping):
    """Identifier for an attack strategy that acts like a dict."""
    attack_type: str
    module: str
    id: str = field(default_factory=lambda: str(uuid.uuid4()))

    def __getitem__(self, key: str) -> str:
        mapping = {
            "__type__": self.attack_type,
            "__module__": self.module,
            "id": self.id
        }
        if key in mapping:
            return mapping[key]
        raise KeyError(key)

    def __iter__(self):
        # Iterate over the keys
        yield "__type__"
        yield "__module__"
        yield "id"

    def __len__(self) -> int:
        return 3
    
    def to_dict(self) -> dict:
        """Convert to a dictionary for JSON serialization."""
        return {
            "__type__": self.attack_type,
            "__module__": self.module,
            "id": self.id
        }

# We need to relate the context type to the result type
class AttackStrategy(ABC, Generic[T_Context, T_Result]):
    """Base class for attack strategies."""

    def __init__(
        self,
        *,
        chat_seed_prompt: Optional[SeedPrompt] = None,
        system_prompt: Optional[SeedPrompt] = None,
        prepend_conversation: Optional[List[SeedPrompt]] = None,
        prompt_normalizer: Optional[PromptNormalizer] = None,
        memory: Optional[MemoryInterface] = None,
    ):
        """Initialize the strategy.
        
        Args:
            adversarial_chat: Chat model that generates attack prompts
            system_prompt: Optional system prompt for the adversarial chat
            memory: Optional memory interface for storing attack data
        """
        self._chat_seed_prompt = chat_seed_prompt
        self._system_prompt = system_prompt
        self._memory = memory or CentralMemory.get_memory_instance()
        self._prepend_conversation = prepend_conversation or []
        self._prompt_normalizer = prompt_normalizer or PromptNormalizer()
        self._memory_labels: dict[str, str] = ast.literal_eval(
            default_values.get_non_required_value(env_var_name="GLOBAL_MEMORY_LABELS") or "{}"
        )
        self._identifier = AttackStrategyIdentifier(
            attack_type=self.__class__.__name__,
            module=self.__module__,
        )

    @property
    def identifier(self) -> dict[str, str]:
        """Get the identifier of the attack strategy."""
        return self._identifier.to_dict()

    @abstractmethod
    async def setup(self, *, context: T_Context) -> None:
        """Prepare the attack strategy."""
        pass
    
    @abstractmethod
    async def execute(self, *, context: T_Context) -> T_Result:
        """
        Execute the attack with the provided context.
        
        Args:
            context: Attack context of type T_Context
        
        Returns:
            Attack result of type T_Result
        """
        pass