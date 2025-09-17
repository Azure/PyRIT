# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import logging
from typing import Optional, TypeVar

from treelib import Tree, Node

from PyRIT.pyrit.memory.central_memory import CentralMemory
from PyRIT.pyrit.prompt_normalizer.prompt_normalizer import PromptNormalizer
from PyRIT.pyrit.prompt_target.common.prompt_chat_target import PromptChatTarget
from pyrit.executor.attack.core import AttackStrategy, AttackContext
from pyrit.models import AttackResult

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
    pass
    
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
    tree: Tree = field(default_factory=Tree())
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

class MultiBranchAttack(AttackStrategy[MultiBranchAttackContextT, MultiBranchResultT]):
    """
    Attach strategy that allows the user to explore multiple branches of an attack tree
    interactively; the full list of commands is defined in MultiBranchCommandEnum.
    
    TODO: Remove scaffolding headers
    """
    
    """ Built-ins (init) """
    
    def __init__(
        self,
        objective_target: PromptChatTarget,
        prompt_normalizer: Optional[PromptNormalizer] = None,
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
        self._prompt_normalizer = prompt_normalizer or PromptNormalizer()
        self._objective_target = objective_target
        
        # Very important detail about context: it should not be initialized here,
        # because there is no objective yet. However, to make this class interactive,
        # we have to have a persistent context that survives multiple _perform_async calls.
        self._ctx = None
        ...
        
    """ Public methods (interface) """
    async def execute_async(self, cmd: CmdT, arg: Optional[str] = None) -> MultiBranchAttack:
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
        self._cmd_handler(cmd, arg)
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
        raise NotImplementedError()
        
    """ Strategy methods (state & exeuction handling) """
    
    # All of these methods use a hack to persist the context between calls.
    # This is because the base class implementation of execute_async assumes
    # that context is mutated without user intervention, which is not the case for
    # this attack.
    
    async def _setup_async(self, *, context):
        self._ctx = context if context else MultiBranchAttackContext()
        return await super()._setup_async(context=context)

    async def _perform_async(self, *, context):
        context = context if context else self._ctx
        return await super()._perform_async(context=context)

    async def _teardown_async(self, *, context):
        context = context if context else self._ctx
        return await super()._teardown_async(context=context)

    """ Helper methods (internal and used only for MultiBranch) """
    def _cmd_handler(self, cmd: CmdT, arg: Optional[str] = None) -> MultiBranchAttackContext:
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
        
        if cmd not in MultiBranchCommandEnum:
            raise ValueError(f"Unknown command: {cmd}")
        
        if arg and not isinstance(arg, str):
            raise ValueError(f"Argument must be a string, got {type(arg)}")
        
        match cmd:
            case CmdT.RETURN:
                self._return_handler(arg)
                message = f"Returned to node {self._ctx.pointer.tag}."
            case CmdT.CONTINUE:
                self._continue_handler(arg)
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
        
        

    
    def _continue_handler(self, arg: str) -> None:
        """
        Handle the CONTINUE command, getting a model response.
        This command creates a new child node with the model's response to the provided
        prompt (arg).
        """
        
        
        
    def _goto_handler(self, arg: str) -> None:
        ...
    
    def _close_handler(self) -> MultiBranchAttackResult:
        """
        Handle the CLOSE command, finalizing the attack and returning the result.
        
        Returns:
            MultiBranchAttackResult: The result of the multi-branch attack.
        """
        ...

    def _new_leaf_handler(self, prompt: str) -> None:
        """
        Handle the NEW_LEAF command, creating a new leaf node in the tree.
        """
        context = self._ctx
        node = Node(
            
        )
        context.tree.add_node(
            response, 
            parent=context.pointer
        )