# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass, field
from enum import Enum
import logging
from typing import Optional, TypeVar

from treelib import Tree, Node

from pyrit.models.attack_result import AttackOutcome
from pyrit.memory.central_memory import CentralMemory
from pyrit.prompt_normalizer.prompt_normalizer import PromptNormalizer
from pyrit.prompt_target.common.prompt_chat_target import PromptChatTarget
from pyrit.executor.attack.core import AttackStrategy, AttackContext, AttackScoringConfig
from pyrit.models import AttackResult, PromptRequestResponse


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
    
    #TODO: Remove debugging fields
    magic_caller: MultiBranchAttack = field(repr=False, compare=False, default=None)
    
    
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
        
        Things that should be supported, but aren't yet:
        * Config objects (constraints on tree size, depth, branching factor, etc.)
        * Prompt converters
        * Replay using a list of commands (interactive is the only way right now)
        * Unit tests (none exist, and this doesn't extend the base AttackStrategy contracts
        faithfully, so it can't)

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
    
    async def execute_async(self, cmd: CmdT, arg: Optional[str] = None) -> MultiBranchAttackResultT:
        """
        Execute a single command in the multi-branch attack strategy.
        
        This method modifies the internal state of the attack context based on the
        command provided. The commands allow navigation and interaction with the
        attack tree.
        
        Args:
            cmd (MultiBranchCommandEnum): The command to execute.
            arg (Optional[str]): An optional argument for the command, such as a response
                                 or a node identifier.
        
        Returns:
            MultiBranchAttack: The updated attack strategy instance.
        
        Raises:
            ValueError: If an invalid command is provided or if required arguments are missing.
        """
        
        # The contract here is that the handler modifies the object in place.
        # Because Python does not require variable assignment on method calls,
        # the user ergonomics are good; we can either call
        # mb_attack = await mb_attack.step_async(cmd, arg)
        # or just
        # await mb_attack.step_async(cmd, arg).
        await self._perform_async(context=self._ctx)
        return self
    
    async def close(self) -> MultiBranchAttackResult:
        """
        Finalize the multi-branch attack and return the result.
        
        This is async because the built-in strategy methods are async, but it does not
        perform any async operations itself; I would like a syncronous version of this,
        or at least an "emergency close" that does not require async.
        
        This method constructs a MultiBranchAttackResult object that encapsulates
        the current state of the attack, including the conversation ID, objective,
        last response, and other relevant details. The result also includes metadata
        about the entire attack tree.
        
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
        return self._intermediate_result_factory(new_context)
        return await self._cmd_handler(cmd=cmd, ctx=context, txt=txt)

    async def _teardown_async(self, *, context):
        return await super()._teardown_async(context=context)
    
    """ Helper methods (internal and used only for MultiBranch) """
    async def _self_intermediate_result_factory(
        self,
        context: MultiBranchAttackContextT
    ) -> MultiBranchAttackResultT:
        return MultiBranchAttackResultT(
            conversation_id=self._ctx.conversation_id,
            objective=self._ctx.objective,
            attack_identifier={"strategy": "multi_branch"},
            last_response=None,
            executed_turns=0,
            execution_time_ms=0, 
            outcome=AttackOutcome.UNDETERMINED, 
            related_conversations=set(),
            metadata={"tree": self._ctx.tree},
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
        You will notice some problems immediately; this is a very rough implementation.
        
        Problem #1: Changes to context are not atomic. Each handler should return a new context,
        but currently they modify self._ctx in place. The default pattern for AttackStrategy is to
        pass around context objects, not modify them in place.
        
        Problem #2: There is little user support. Error messages and handling are really basic.
        
        Problem #3: There is no support for concurrency. This is a single-threaded implementation,
        and it is not thread-safe, because the context object is modified in place, and it is directly
        attached to the class instance.
        
        Problem #4: Nodes are identified with their identifiers, but we only show the tag to the user.
        This means that users have to remember the mapping between tags and identifiers, which is not
        user-friendly. We should have a way to look up nodes by their tags, or show identifiers instead.
        
        Args:
            cmd (MultiBranchCommandEnum): The command to parse.
            arg (Optional[str]): An optional argument for the command.
        
        Returns:
            MultiBranchAttackContext: The updated attack context after executing the command.
        """
        
        intermediate_result: MultiBranchAttackResultT = self._self_intermediate_result_factory(ctx)   

        
        if cmd not in MultiBranchCommandEnum:
            raise ValueError(f"Unknown command: {cmd}")
        
        if arg and not isinstance(arg, str):
            raise ValueError(f"Argument must be a string, got {type(arg)}")
        
        match cmd:
            case CmdT.RETURN:
                result = MultiBranchAttackResultT(
                    self._return_handler(arg)
                )
                message = f"Returned to node {self._ctx.pointer.tag}."
            case CmdT.CONTINUE:
                response = await self._continue_handler(arg)
                message = f"Continued and added node {self._ctx.pointer.tag}."
            case CmdT.GOTO:
                self._goto_handler(arg)
                message = f"Moved to node {self._ctx.pointer.tag}."
            case CmdT.CLOSE:
                print("WARNING: Closing the attack will return the final result" \
                    "and not the attack object. Call `result = await mb_attack.close()`" \
                    "if this is what you want. (This command has not changed the state of the attack.)")
            case _:
                raise ValueError(f"Unknown command: {cmd}")
            
        print(message)
        print(f"Current tree state: {self._ctx.tree.show()}")
                          
    def _return_handler(self, arg: str) -> None:
        """
        Handle the RETURN command, moving the pointer to the parent node.
        
        Args:
            arg (str): Not used for this command.
        """
        context = self._ctx
        if context.pointer == "root":
            raise ValueError("Already at root node; cannot return to parent.")
        parent = context.tree.parent(context.pointer)
        if parent is None:
            raise ValueError("Current node has no parent; cannot return.")
        context.pointer = parent.identifier
        
    async def _continue_handler(self, arg: str) -> None:
        """
        Handle the CONTINUE command, getting a model response.
        This command creates a new child node with the model's response to the provided
        prompt (arg).
        """
        response = await self._objective_target.send_prompt_async(arg)
        self._new_leaf_handler(response)
        return response

    def _goto_handler(self, arg: str) -> None:
        """
        Handle the GOTO command, moving the pointer to a specified node by its tag.
        """
        target = self._ctx.tree.get_node_by_tag(arg)
        if target is None:
            raise ValueError(f"Unknown node tag: {arg}")
        self._ctx.pointer = target.identifier

    async def _close_handler(self) -> MultiBranchAttackResult:
        """
        Handle the CLOSE command, finalizing the attack and returning the result.
        
        Returns:
            MultiBranchAttackResult: The result of the multi-branch attack.
        """
    
        return await self._perform_async(context=self._ctx)

    def _new_leaf_handler(self, prompt_request_response: PromptRequestResponse) -> None:
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