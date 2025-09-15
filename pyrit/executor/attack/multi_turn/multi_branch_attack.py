from __future__ import annotations

from treelib import Node, Tree
from enum import Enum
from dataclasses import dataclass, field
from abc import ABC
from typing import AsyncIterable, Optional, TypeVar, Generator

from pyrit.executor.attack.core import (
    AttackContext,
    AttackStrategy,
    AttackStrategyResultT,
)

MultiBranchAttackContextT = TypeVar("MultiBranchAttackContextT", bound="MultiBranchAttackContext")

class MultiBranchCommand(Enum):
    AUTOCOMPLETE = "autocomplete"
    BRANCH = "branch"
    RESPOND = "respond"
    BACK = "back"
    FORWARD = "forward"
    CLOSE = "close"

@dataclass
class MultiBranchAttackContext(AttackContext):
    """Context for multi-branch attacks"""

    # Tree structure to hold the branches of the attack
    attack_tree: Tree = field(default_factory=lambda: Tree())

    # Current node in the attack tree
    current_node: Optional[Node] = None
    
    # Cache for all leaves of the tree with their respective conversation IDs
    leaves_cache: dict[str, Node] = field(default_factory=dict)
    
    

class MultiBranchAttack(
    AttackStrategy[MultiBranchAttackContextT, AttackStrategyResultT], 
    ABC
):
    """
    Attack for executing multi-branch attacks.
    """
    
    def __init__(self, **kwargs):
        """
        Initialize the multi-branch attack strategy.
        """
        super().__init__(**kwargs)

    async def execute_async_as_step(self) -> MultiBranchAttack:
        """
        To give the user full control of the attack flow, we return slightly mutated instances of the
        MultiBranchAttack object. 
        """
        raise NotImplementedError()

    async def step(self, cmd: MultiBranchCommand, txt: str) -> MultiBranchAttack | AttackStrategyResultT:
        match cmd:
            case MultiBranchCommand.AUTOCOMPLETE:
                return self._autocomplete_handler()
            case MultiBranchCommand.BRANCH:
                return self._branch_handler(txt)
            case MultiBranchCommand.RESPOND:
                return self._respond_handler(txt)
            case MultiBranchCommand.BACK:
                return self._back_handler()
            case MultiBranchCommand.FORWARD:
                return self._forward_handler(txt)
            case MultiBranchCommand.CLOSE:
                return self._close_handler()
            case _:
                raise ValueError(f"Unknown command: {cmd}")
        
    def __repr__(self):
        # Retrieve current path and format it before returning.
        raise NotImplementedError()
    
    def _autocomplete_handler(self) -> None:
        raise NotImplementedError()
    
    def _branch_handler(self, branch_text: str) -> None:
        raise NotImplementedError()
    
    def _respond_handler(self, response_text: str) -> None:
        raise NotImplementedError()
    
    def _back_handler(self) -> None:   
        raise NotImplementedError
    
    def _forward_handler(self) -> None:
        raise NotImplementedError
    
    def _close_handler(self) -> AttackStrategyResultT:
        raise NotImplementedError()
        """
        1. Score final result
        2. Close all active conversations
        3. Validate result formmating
        4. Return to caller function (step)
        """