# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass, field
from enum import Enum
import logging
from typing import Optional, TypeVar, overload

from treelib import Tree, Node

from PyRIT.build.lib.pyrit.models.prompt_request_piece import PromptRequestPiece
from pyrit.memory.central_memory import CentralMemory
from pyrit.prompt_normalizer.prompt_normalizer import PromptNormalizer
from pyrit.prompt_target.common.prompt_chat_target import PromptChatTarget
from pyrit.executor.attack.core import(
    AttackContext,
    AttackStrategy,
    AttackScoringConfig
)
    
from pyrit.models import (
    AttackOutcome,
    AttackResult, 
    PromptRequestResponse,
)


MultiBranchAttackContextT = TypeVar("MultiBranchAttackContextT", bound="MultiBranchAttackContext")
MultiBranchAttackResultT = TypeVar("MultiBranchResultT", bound="MultiBranchAttackResult")
CmdT = TypeVar("CmdT", bound="MultiBranchCommandEnum")

logger = logging.getLogger(__name__)

@dataclass
class MultiBranchAttackResult(AttackResult):
    """
    The multibranch attack result is basically a wrapper for the AttackResult base
    class, but with a tree structure in the metadata for reference.
    
    Eventually, I would like to extend this into a composition of other AttackResults,
    such that each leaf node in the tree has its own AttackResult object; this lets us
    extend the base case by using the winning conversation for the fields like
    conversation_id without losing the rest of the tree.
    
    Very crucially, calling close() on the MultiBranchAttack class will populate
    this tree with ONLY ONE CONVERSATION PATH, the one currently pointed to.
    
    Everything else is stashed in metadata but is otherwise lost.
    """
    
    
@dataclass
class MultiBranchAttackContext(AttackContext):
    """
    This wraps the attack context class and contains the current state of the
    attack through the tree.
    
    It is currently defined as a property of the MultiBranchAttack class; this
    behavior is not desired, because it couples state with strategy. However,
    the current implementation of execute_async() assumes all context updates occur
    in non-interactive attacks, so this is a workaround that persists until a refactor.
    
    Ideally, we would subclass the node class to provide some guarantees about its contents.
    As-is, each node is expected to use its properties like this:
    
    identifier (str): Id for data storage and navigation, unique.
    tag (str): Given a random letter to make it easier for operators to navigate.
    expanded (bool): Controls visibility of children in tree displays. Unchanged.
    data (Any): A dict containing the PromptRequestResponse associated with it.

    For validation, the root node should have as many PromptRequestResponses as there
    are leaves, because each leaf represents a complete conversation.
    """
    objective: str = None # Attack objective
    tree: Tree = field(default_factory=Tree)
    pointer: str = "root" # Current node identifier
    cmd: Optional[CmdT] = None # Last command
    arg: Optional[str] = None # Last argument
    
class MultiBranchCommandEnum(Enum):
    """
    All possible commands that can be executed at each step of the multibranch attack.
    The user is resquired to provide the correct arguments for each command; there is
    currently no validation or help menu provided.
    """
    RETURN = "return" # Return to the parent (previous) node.
    CONTINUE = "continue" # Provide a response to the model. By default, this branches.
    GOTO = "goto" # Go to a specific node by its name (label).

class MultiBranchAttack(AttackStrategy[MultiBranchAttackContextT, AttackResult]):
    """
    Attach strategy that allows the user to explore multiple branches of an attack tree
    interactively; the full list of commands is defined in MultiBranchCommandEnum.
    
    TODO: Remove scaffolding headers
    """
    
    """ Built-ins (init) """
    
    def __init__(
        self,
        objective_target: PromptChatTarget,
        objective: str,
        prompt_normalizer: Optional[PromptNormalizer] = None,
        scoring_config: Optional[AttackScoringConfig] = None,
    ):
        """
        Implementation of the multi-branch attack strategy, an interactive strategy where
        users can explore different branches of an attack tree.
        
        Usage Pattern:
        >>> attack = MultiBranchAttack(objective_target, objective)
        >>> result = await attack.step(MultiBranchCommandEnum.CONTINUE, "Prompt text")
        >>> result = 
        >>> result = await attack.close()
        
        alternatively,
        
        ```
        attack = MultiBranchAttack(objective_target, objective)
        while result.context:
            command, argument = get_user_input()  # Pseudo-function to get user input
            result = await attack.step(command, argument)
        ```

        Commands Available:
        - RETURN: Move back to the parent node in the tree. No argument needed.
          ex: result = await attack.step(MultiBranchCommandEnum.RETURN)
        - CONTINUE: Add a new response to the current node and branch out. Requires a
                    string argument representing the user input.
          ex: result = await attack.step(MultiBranchCommandEnum.CONTINUE, "User response")
        - GOTO: Move to a specific node in the tree by its tag. Requires a string argument
                representing the node's tag.
          ex: result = await attack.step(MultiBranchCommandEnum.GOTO, "A")
        
        To finish the attack and get the final result, call the `close()` method.
            
       
        Attributes:
            objective_target: The target model for the attack.
            prompt_normalizer: Optional prompt normalizer to use for the attack.
            memory: Standard central memory instance for storing conversations.
            ctx: The current context of the attack. This is in its current state a magical
            property of the class, but it should be passed around like in other strategies,
            eventually.
            
        Args:
            objective_target: The target model for the attack.
            prompt_normalizer: Optional prompt normalizer to use for the attack.
        
        Returns:
            None.
        """
        
        super().__init__(logger=logger, context_type=MultiBranchAttackContext)
        
        self._memory = CentralMemory.get_memory_instance()
        
        self._objective = objective
        self._prompt_normalizer = prompt_normalizer or PromptNormalizer()
        self._objective_target = objective_target
        self._objective_scorer = scoring_config.objective_scorer if scoring_config else AttackScoringConfig()
        
        
    """ Public methods (interface) """
    
    async def step(
        self, 
        cmd: CmdT, 
        prev: MultiBranchAttackResultT,
        arg: Optional[str] = None
    ) -> MultiBranchAttackResultT:
        """
        Execute a single command in the multi-branch attack strategy.
        
        Args:
            cmd (MultiBranchCommandEnum): The command to execute.
            arg (Optional[str]): An optional argument for the command, such as a response
                                 or a node identifier.
        
        Returns:
            MultiBranchAttack: The updated attack strategy instance.
        
        Raises:
            ValueError: If an invalid command is provided or if required arguments are missing.
        """

        prev.context.cmd = cmd
        prev.context.arg = arg
        return await self.execute_async(context=prev)


    async def close(self) -> MultiBranchAttackResult:
        """
        Finalize the multi-branch attack and return the result.
        
        Returns:
            MultiBranchAttackResult: The result of the multi-branch attack.
        """
        return await self._close_handler()

    """ Lifecycle Methods (from Strategy base class) """

    async def _validate_context(self, *, context):
        if not context.objective:
            raise ValueError("Objective must be provided for MultiBranchAttack.")   
    
    async def _setup_async(self, *, context):
        context = context if context else MultiBranchAttackContext()
        print(f"Setting up MultiBranchAttack with objective: {self._objective}")
        print("Read this class docstring for a list of commands and usage patterns.")

    async def _perform_async(
        self,
        context: MultiBranchAttackContextT
    ) -> MultiBranchAttackResultT:
        cmd, arg = context.cmd, context.arg
        new_context = await self._cmd_handler(cmd=cmd, ctx=context, arg=arg)
        return await self._intermediate_result_factory(new_context)

    async def _teardown_async(self, *, context):
        return await super()._teardown_async(context=context)
    
    """ Helper methods (internal and used only for MultiBranch) """
    async def _self_intermediate_result_factory(
        self,
        context: MultiBranchAttackContextT
    ) -> MultiBranchAttackResultT:

        return MultiBranchAttackResultT(
            conversation_id=context.pointer.id,
            objective=context.objective,
            attack_identifier={"strategy": "multi_branch"},
            last_response=context.pointer.data.get("responses", [])[-1] if context.pointer.data else None,
            executed_turns=len(context.tree.path) - 1,
            execution_time_ms=0, 
            outcome=AttackOutcome.UNDETERMINED, 
            related_conversations=set(),
            metadata={"tree": context.tree},
            context=context
        )
    
    async def _cmd_handler(
        self, 
        cmd: CmdT, 
        ctx: MultiBranchAttackContext,
        arg: Optional[str] = None
    ) -> MultiBranchAttackContext:
        """
        Parse the command and its arguments for execution.

        Args:
            cmd (MultiBranchCommandEnum): The command to parse.
            ctx (MultiBranchAttackContext): The current attack context.
            arg (Optional[str]): An optional argument for the command.
        
        Returns:
            MultiBranchAttackContext: The updated attack context after executing the command.
        """
        
        if cmd not in MultiBranchCommandEnum:
            raise ValueError(f"Unknown command: {cmd}")
        
        if arg and not isinstance(arg, str):
            raise ValueError(f"Argument must be a string, got {type(arg)}")
        
        match cmd:
            case CmdT.RETURN:
                parent_tag = ctx.tree.parent(ctx.pointer).tag if ctx.pointer != "root" else None
                if not parent_tag:
                    raise ValueError("Already at root node; cannot return to parent.")
                ctx = await self._goto_handler(ctx, tag=parent_tag)
                message = f"Returned to node {self._ctx.pointer.tag}."
            case CmdT.CONTINUE:
                ctx = await self._continue_handler(ctx, tag=arg)
                message = f"Continued and added node {self._ctx.pointer.tag}."
            case CmdT.GOTO:
                ctx = await self._goto_handler(ctx, tag=arg)
                message = f"Moved to node {self._ctx.pointer.tag}."
            case CmdT.CLOSE:
                ctx = ctx # No-op, handled elsewhere.
                message = f"To close the attack, call .close() instead. This command has not changed the state."
            case _:
                raise ValueError(f"Unknown command: {cmd}")
            
        print(message)
        print(f"Current tree state: {self._ctx.tree.show()}")
        intermediate_result: MultiBranchAttackResultT = self._self_intermediate_result_factory(ctx)   
        return intermediate_result
    
    async def _goto_handler(context: MultiBranchAttackContext, tag: Optional[str]) -> MultiBranchAttackContext:
        """
        Handle the RETURN command to navigate to the parent node.

        Args:
            context (MultiBranchAttackContext): The current attack context.
            arg (Optional[str]): Not used for this command.

        Returns:
            MultiBranchAttackContext: The updated attack context after returning to the parent node.
        """
        # Validate that tag is in tree first
        all_tags = [node.tag for node in context.tree.all_nodes()]
        if tag and tag not in all_tags:
            raise ValueError(f"Node with tag {tag} does not exist in the tree.")
        context.pointer = tag
        return context
    
    async def _continue_handler(context: MultiBranchAttackContext, command: Optional[str]) -> MultiBranchAttackContext:
        """
        Get a model response and get a completion from the model target.
        """
        