# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass, field
from enum import Enum
import logging
from typing import Optional, TypeVar

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
    objective: str = None
    tree: Tree = field(default_factory=Tree)
    pointer: str = "root" # This is the node identifier.
    
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
        
        # Very important note about central memory: it is called very often here, because
        # each leaf node adds a conversation id and as many PromptRequestResponses as the
        # current path is deep.
        
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

            
        return await self.execute_async(cmd=cmd, context=prev, arg=arg)
    
    async def execute_async(
        self, 
        cmd: CmdT,
        context: MultiBranchAttackContext = None,
        arg: Optional[str] = None
    ) -> AttackResult:
        """
        """
        return await super().execute_async(context=context)

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

    async def _perform_async(
        self, 
        cmd: CmdT,
        txt: str,
        context: MultiBranchAttackContextT
    ) -> MultiBranchAttackResultT:
        new_context = await self._cmd_handler(cmd=cmd, ctx=context, arg=txt)
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
                ctx = await self._return_handler(ctx, arg)
                message = f"Returned to node {self._ctx.pointer.tag}."
            case CmdT.CONTINUE:
                ctx = await self._continue_handler(ctx, arg)
                message = f"Continued and added node {self._ctx.pointer.tag}."
            case CmdT.GOTO:
                ctx = await self._goto_handler(ctx, arg)
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

    async def _return_handler(self, ctx: MultiBranchAttackContext, arg: str) -> None:
        """
        Handle the RETURN command, moving the pointer to the parent node.
        
        Args:
            arg (str): Not used for this command.
        """
        if ctx.pointer == "root":
            raise ValueError("Already at root node; cannot return to parent.")
        parent = ctx.tree.parent(ctx.pointer)
        if parent is None:
            raise ValueError("Current node has no parent; cannot return.")
        ctx.pointer = parent.identifier
        return ctx

    async def _continue_handler(self, ctx: MultiBranchAttackContext, arg: str) -> MultiBranchAttackContext:
        """
        Handle the CONTINUE command, getting a model response.
        This command creates a new child node with the model's response to the provided
        prompt (arg).
        """
        user_input: PromptRequestResponse = self._create_user_input(arg)
        model_response: PromptRequestResponse = await self._objective_target.send_prompt_async(user_input)
        return self._new_leaf_handler(user_input, model_response, ctx)
    
    def _goto_handler(self, arg: str) -> None:
        """
        Handle the GOTO command, moving the pointer to a specified node by its tag.
        """
        target = self._ctx.tree.get_node_by_tag(arg)
        if target is None:
            raise ValueError(f"Unknown node tag: {arg}")
        self._ctx.pointer = target.identifier

    def _create_user_input(self, prompt: str) -> PromptRequestResponse:
        """
        Create a PromptRequestResponse object from the user's input prompt.
        
        Args:
            prompt (str): The user's input prompt.
        
        Returns:
            PromptRequestResponse: The constructed PromptRequestResponse object.
        """
        if not prompt or not isinstance(prompt, str):
            raise ValueError("Prompt must be a non-empty string.")
        
        conversation_id = self._ctx.conversation_id or default_values.DEFAULT_CONVERSATION_ID
        user_input = PromptRequestResponse(
            request_pieces=[
                PromptRequestPiece(
                    role="user",
                    conversation_id=conversation_id,
                    original_value=prompt,
                    converted_value=self._prompt_normalizer.normalize(prompt),
                    prompt_target_identifier=self._objective_target.get_identifier(),
                )
            ]
        )
        return user_input

    async def _close_handler(self) -> MultiBranchAttackResult:
        """
        Handle the CLOSE command, finalizing the attack and returning the result.
        
        Returns:
            MultiBranchAttackResult: The result of the multi-branch attack.
        """
    
        return await self._perform_async(context=)

    
    def _new_leaf_handler(
        self, 
        user_input: PromptRequestResponse,
        model_output: PromptRequestResponse,
        ctx: MultiBranchAttackContext
    ) -> MultiBranchAttackContext:
        """
        Handle the creation of a new leaf after CONTINUE is executed.
        """
        
        # 1 First extract the unique conversation ID from the PromptRequestResponse
        conversation_id = prompt_request_response.conversation_id
        
        # 2 Then create the node and assign it a tag (letter) and the prompt request response
        node = Node(
            tag=chr(65 + len(self._ctx.tree.nodes)),
            data={
                "responses": [prompt_request_response]
            }
        )
        self._ctx.tree.add_node(node, parent=self._ctx.pointer)
        self._ctx.pointer = node.identifier
        logger.debug(f"Added new node {node.tag} with conversation ID {conversation_id}")
        
        # 3 Add all parent nodes to the conversation in memory using the new conversation ID
        
        full_path = [ancestor for ancestor in self._ctx.tree.rsearch(self._ctx.pointer)]
        for ancestor in full_path:
            new_prr = PromptRequestResponse(
                conversation_id=conversation_id,
                requests=ancestor.data.requests if ancestor.data else [],
                responses=ancestor.data.responses if ancestor.data else [],
            )
            self._memory.add_request_response_to_memory(new_prr)