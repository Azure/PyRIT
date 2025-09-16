# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import asyncio
import logging
import uuid

from __future__ import annotations
from abc import ABC
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, TypeVar

from treelib.tree import Node, Tree

from pyrit.common.path import DATASETS_PATH
from pyrit.common.utils import combine_dict, warn_if_set
from pyrit.exceptions import (
    InvalidJsonException,
    pyrit_json_retry,
    remove_markdown_json,
)
from pyrit.executor.attack.core import (
    AttackAdversarialConfig,
    AttackContext,
    AttackConverterConfig,
    AttackScoringConfig,
    AttackStrategy,
    AttackStrategyResultT,
)
from pyrit.memory import CentralMemory
from pyrit.models import (
    AttackOutcome,
    AttackResult,
    ConversationReference,
    ConversationType,
    PromptRequestPiece,
    PromptRequestResponse,
    Score,
    SeedPrompt,
    SeedPromptGroup,
)
from pyrit.prompt_normalizer import PromptConverterConfiguration, PromptNormalizer
from pyrit.prompt_target import PromptChatTarget
from pyrit.score import (
    Scorer,
    SelfAskScaleScorer,
    SelfAskTrueFalseScorer,
    TrueFalseQuestion,
)

logger = logging.getLogger(__name__)

MultiBranchAttackContextT = TypeVar("MultiBranchAttackContextT", bound="MultiBranchAttackContext")

class MultiBranchCommand(Enum):
    """
    All possible commands that can be executed in a multi-branch attack.
    You can think of this as the possible states of the attack object, where
    the handler is a transition function between states.
    """
    AUTOCOMPLETE = "autocomplete"
    BRANCH = "branch"
    RESPOND = "respond"
    BACK = "back"
    FORWARD = "forward"
    CLOSE = "close"


@dataclass
class MultiBranchAttackResult(AttackResult):
    """Result of a multi-branch attack"""
    pass
    # fields populated by metadata attribute of attackresult?

@dataclass(frozen=True)
class MultiBranchAttackConfig:
    max_depth: int = field(default=10)
    max_children: int = field(default=3)
    max_leaves: int = field(default=20)
    live: bool = field(default=False)


@dataclass
class MultiBranchAttackContext(AttackContext):
    """
    Context for multi-branch attacks.
    
    Parameters 
    """

    # Tree structure to hold the branches of the attack
    attack_tree: Tree = field(default_factory=lambda: Tree())

    # Current node in the attack tree
    current_node: Optional[Node] = None
    
    # Current conversation_id
    current_conversation_id: Optional[str] = None

    # Cache for all leaves of the tree with their respective conversation IDs
    leaves_cache: dict[str, Node] = field(default_factory=dict)


class MultiBranchAttack(AttackStrategy[MultiBranchAttackContextT, AttackStrategyResultT], ABC):
    """
    Attack for executing multi-branch attacks.
    """

    def __init__(
        self,
        *,
        objective_target: PromptChatTarget,
        attack_config: MultiBranchAttackConfig,
        
    ):
        """
        Initialize the multi-branch attack strategy.
        
        Args:
        
        """
        super().__init__(context_type=MultiBranchAttackContext, logger=logger)
        
        self._memory = CentralMemory.get_instance()
        
        
    
    async def send_prompt_async(self, objective: str) -> None:
        """
        Unlike other implementations of send_prompt_async, this one is secondary to the
        built-in step method. It is run in two cases:
        1. When the attack is set up for the very first time (send_prompt assigns the root node of 
            config)
        2. When the user issues a command that requires a new prompt to be sent to the model
        (e.g., RESPOND, BRANCH).
        
        Args:
        """
        ...
        
        

    async def step(self, cmd: MultiBranchCommand, txt: str | None) -> MultiBranchAttack:
        self._validate_command(cmd, txt)
        match cmd:
            case MultiBranchCommand.AUTOCOMPLETE:
                self._autocomplete_handler()
            case MultiBranchCommand.BRANCH:
                if not txt:
                    print(
                        "WARNING: Branch command requires non-empty text. (This command has not changed the state of the attack.)"
                    )
                self._branch_handler(txt)
            case MultiBranchCommand.RESPOND:
                self._respond_handler(txt)
            case MultiBranchCommand.BACK:
                self._back_handler()
            case MultiBranchCommand.FORWARD:
                self._forward_handler(txt)
            case MultiBranchCommand.CLOSE:
                print(
                    "WARNING: Closing the attack will return the final result and not the attack object."
                    "If this is what you want, run `result = await mb_attack.close()` instead."
                    "(This command has not changed the state of the attack.)"
                )
            case _:
                raise ValueError(f"Unknown command: {cmd}")

        return self

    def close(self) -> AttackStrategyResultT:
        """
        Finalize the attack and return the result.
        """
        
        """
        1. Score final result
        2. Close all active conversations
        3. Validate result formmating
        4. Return to caller function (step)
        """
        # validation - is attack ready to be closed?
        
        result = MultiBranchAttackResult(
            conversation_id=self.context.adversarial_chat_conversation_id,
            objective=self.context.objective,
            attack_identifier={"name": "multi_branch_attack"},
            executed_turns=self.context.turn_count,
            execution_time_ms=self.context.get_execution_time_ms(),
            outcome=AttackOutcome.UNDETERMINED,
            outcome_reason="Not implemented",
            total_branches=len(self.context.attack_tree.nodes) - 1,  # Exclude root
            max_depth=self.context.attack_tree.depth(),
            total_leaves=len(self.context.leaves_cache),
        )
        
        return result

    def _validate_command(self, cmd: MultiBranchCommand, txt: str | None) -> None:
        if not isinstance(cmd, MultiBranchCommand):
            raise ValueError(f"Invalid command: {cmd}")
        if not isinstance(txt, (str, type(None))):
            raise ValueError(f"Text must be a string or None, got: {type(txt)}")
        
    def __repr__(self):
        # Retrieve current path and format it before returning.
        raise NotImplementedError()

    def _autocomplete_handler(self) -> None:
        raise NotImplementedError()

    def _branch_handler(self, branch_text: str) -> None:
        # Add node
        # Check if 
        raise NotImplementedError()

    def _respond_handler(self, response_text: str) -> None:
        # Get model response and add to current conversation
        raise NotImplementedError()

    def _back_handler(self) -> None:
        # Change current pointer to parent node.
        raise NotImplementedError

    def _forward_handler(self) -> None:
        raise NotImplementedError