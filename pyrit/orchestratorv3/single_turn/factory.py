from __future__ import annotations
from enum import Enum
from typing import overload

from pyrit.orchestratorv3.single_turn.base import SingleTurnBaseOrchestratorBuilder
from pyrit.orchestratorv3.single_turn.prompt_sending import PromptSendingOrchestratorBuilder
from pyrit.prompt_target.common.prompt_target import PromptTarget

class SingleTurnOrchestratorType(Enum):
    """Types of single-turn orchestrators."""
    PROMPT_SENDING = "prompt_sending"

class SingleTurnOrchestratorFactory:
    """Factory for creating single-turn orchestrator builders."""
    
    @overload
    @classmethod
    def create(cls, target: PromptTarget, 
               builder_type: "SingleTurnOrchestratorType.PROMPT_SENDING", 
               **kwargs) -> PromptSendingOrchestratorBuilder: ...
    
    @classmethod
    def create(cls, target: PromptTarget, 
               builder_type: SingleTurnOrchestratorType = SingleTurnOrchestratorType.PROMPT_SENDING, 
               **kwargs) -> SingleTurnBaseOrchestratorBuilder:
        """Create a new single-turn orchestrator builder.
        
        Args:
            target: Target system to attack
            builder_type: Type of orchestrator to create
            
        Returns:
            A configured builder extending SingleTurnBaseOrchestratorBuilder
        """
        if not target:
            raise ValueError("Target must be provided")
            
        match builder_type:
            case SingleTurnOrchestratorType.PROMPT_SENDING:
                return PromptSendingOrchestratorBuilder(target=target, **kwargs)
            case _:
                raise ValueError(f"Unknown builder type: {builder_type}")