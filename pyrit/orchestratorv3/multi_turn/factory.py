from __future__ import annotations
from enum import Enum
from typing import TypeVar, overload
from pyrit.orchestratorv3.multi_turn.base import MultiTurnBaseOrchestratorBuilder
from pyrit.orchestratorv3.multi_turn.red_teaming import RedTeamingOrchestratorBuilder
from pyrit.prompt_target.common.prompt_chat_target import PromptChatTarget
from pyrit.prompt_target.common.prompt_target import PromptTarget


class MultiTurnOrchestratorType(Enum):
    """Types of multi-turn orchestrators."""
    RED_TEAMING = "red_teaming"

class MultiTurnOrchestratorFactory:
    """Factory for creating multi-turn orchestrator builders."""
    
    @overload
    @classmethod
    def create(cls, target: PromptTarget, chat: PromptChatTarget, 
               builder_type: "MultiTurnOrchestratorType.RED_TEAMING", 
               **kwargs) -> RedTeamingOrchestratorBuilder: ...
    
    @classmethod
    def create(cls, target: PromptTarget, chat: PromptChatTarget, 
               builder_type: MultiTurnOrchestratorType = MultiTurnOrchestratorType.RED_TEAMING, 
               **kwargs) -> MultiTurnBaseOrchestratorBuilder:
        """Create a new multi-turn orchestrator builder."""
        if not target or not chat:
            raise ValueError("Target and chat model must be provided")
            
        match builder_type:
            case MultiTurnOrchestratorType.RED_TEAMING:
                return RedTeamingOrchestratorBuilder(target=target, adversarial_chat=chat)
            case _:
                raise ValueError(f"Unknown builder type: {builder_type}")

