# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
from __future__ import annotations

import asyncio
import logging
import uuid

from abc import ABC
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, TypeVar, Self, Union

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
CmdT = TypeVar("CmdT", bound="MultiBranchCommand")

class MultiBranchCommand(Enum):
    """
    All possible commands that can be executed in a multi-branch attack.
    You can think of this as the possible states of the attack object, where
    the handler is a transition function between states.
    
    """
    UP = "up" # Move to parent node
    DOWN = "down" # Move to a child node (if multiple children, specify which.
    # (if no children are specified, create a new one)
    CLOSE = "close" # End the attack


@dataclass
class MultiBranchAttackResult(AttackResult):
    """Result of a multi-branch attack"""
    # add a history of commands executed
    pass
    # fields populated by metadata attribute of attackresult?

@dataclass(frozen=True)
class MultiBranchAttackConfig:
    max_depth: int = field(default=10)
    max_children: int = field(default=3)
    max_leaves: int = field(default=20)
    live: bool = field(default=False)


@dataclass
class MultiBranchAttackNode(Node):
    parent_id: Optional[str] = None
    prompt_request_response: Optional[PromptRequestResponse] = None

@dataclass
class MultiBranchAttackContext(AttackContext):
    """
    Context for multi-branch attacks.
    
    Parameters 
    """

    # Tree structure to hold the branches of the attack
    attack_tree: Tree = field(default_factory=lambda: Tree(node_class=MultiBranchAttackNode))

    # Current node in the attack tree
    current_node: Optional[Node] = field(default=attack_tree.get_node("root"))
    
    # Current conversation_id
    current_conversation_id: Optional[str] = field(default=None)

    # Cache for all leaves of the tree with their respective conversation IDs
    leaves_cache: dict[str, Node] = field(default_factory=dict)
    
    # Timesteps
    turn_count: int = field(default=0)
    
    # Actions taken so far (for undo functionality)
    # actions: list[tuple[MultiBranchCommand, Optional[str]]] = field(default_factory=list)
    
    


class MultiBranchAttack(AttackStrategy[MultiBranchAttackContextT, AttackStrategyResultT], ABC):
    """
    Attack for executing multi-branch attacks.
    """
    
    """
    Built-ins
    """    

    def __init__(
        self,
        *,
        objective_target: PromptChatTarget,
        attack_config: MultiBranchAttackConfig,
        scoring_config: Optional[AttackScoringConfig] = None,
        prompt_normalizer: Optional[PromptNormalizer] = None,      
    ):
        """
        Initialize the multi-branch attack strategy.
        
        Args:
            objective_target (PromptChatTarget): The target model for the attack.
            attack_config (MultiBranchAttackConfig): Configuration for the multi-branch attack.
            prompt_normalizer (Optional[PromptNormalizer]): Optional prompt normalizer to preprocess prompts.
        
        Raises:
            ValueError: If any of the configuration parameters are invalid.
        """
        super().__init__(context_type=MultiBranchAttackContext, logger=logger)
        
        self._memory = CentralMemory.get_memory_instance()
        self.context = None
        
    def __repr__(self):
        # Retrieve current path and format it before returning.
        raise NotImplementedError()
    
    """
    Public methods
    """
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
        
        

    async def step(self, cmd: CmdT, txt: Union[str, None] = None) -> Self:
        """
        
        """
        if self.context is None:
            raise ValueError("Context is not initialized. Please run execute_async first.")
        self._validate_command(cmd, txt)
        match cmd:
            case MultiBranchCommand.UP:
                """
                self._pointer.get_parent()
                config.change_state()
                return
                """
                ...
            case MultiBranchCommand.DOWN:
                """
                if
                """    
            case MultiBranchCommand.CLOSE:
                print(
                    "WARNING: Closing the attack will return the final result and not the attack object."
                    "If this is what you want, run `result = await mb_attack.close()` instead."
                    "(This command has not changed the state of the attack.)"
                )
            case _:
                raise ValueError(f"Unknown command: {cmd}")

        return self

    async def execute_async(self, objective: str) -> Self:
        """
        
        """
        # raise valueerror if commands are not properly formatted
        # parse full commands list first
        # if commands: run commands start to finish
        # else: enter while loop and each input -> step until close

        if self.context is None:
            await self._setup_async(context=MultiBranchAttackContext())
            self.context.objective = objective
            
        print(f"Starting multi-branch attack with objective: {objective}")
        print("Use the `step` method to interact with the attack.")
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
        self._teardown_async(context=self.context)
        
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

    """ Private helper methods (not inherited) """

    def _validate_command(self, cmd: MultiBranchCommand, txt: str | None) -> None:
        if not isinstance(cmd, MultiBranchCommand):
            raise ValueError(f"Invalid command: {cmd}")
        if not isinstance(txt, (str, type(None))):
            raise ValueError(f"Text must be a string or None, got: {type(txt)}")
        

    
    def _new_leaf_handler(self) -> None:
        """
        When a new leaf is added to the tree (check during step)
        We call this handler to add a new conversation id
        And also update the memory instance with the new path
        (A) -> (B) -> (C)
        + (B) -> (D)
        new conversation_id FOOBAR
        
        Add to memory (wrap each as PromptRequestResponse):
        (A)-FOOBAR 
        (B)-FOOBAR
        (D)-FOOBAR
        """
        raise NotImplementedError()

    
    """ Private lifecycle management methods from AttackStrategy """
    def _validate_context(self, *, context):
        if not isinstance(context, MultiBranchAttackContext):
            raise ValueError(f"Context must be of type MultiBranchAttackContext, got: {type(context)}")
        return super()._validate_context(context=context)
    
    def _setup_async(self, *, context):
        """
        
        """
        return super()._setup_async(context=context)
    
    
    def _teardown_async(self, *, context):
        """
        Free all memory and make sure result object is well-placed
        """
        return super()._teardown_async(context=context)
    
    def _perform_async(self, *, context):
        """
        State management for multi-branch attack, called on step
        """
        return super()._perform_async(context=context)
    