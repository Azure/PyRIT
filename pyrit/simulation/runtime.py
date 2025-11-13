# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

from dataclasses import dataclass
from pyrit.models.message import Message
from pyrit.models.message_piece import MessagePiece

class Runtime:
    def __init__(self) -> None:
        """
        The runtime object is a singleton that enables a PromptTarget to access a simulated
        environment, e.g. a Docker container.
        >>> PromptTarget._runtime = <runtime_object at xxxx>
        >>> PromptTarget.send_prompt_async calls PromptTarget._runtime.send_async to parse messages.
        """
        raise NotImplemented
        
    async def send_async(self, command: Message) -> Message:
        """
        Process the target input (called from PromptTarget) into a result.
        """
        raise NotImplemented
    
    async def _initialize_container(self) -> Container:
        """
        Create new container.
        """
        raise NotImplemented
    

class Container:
    def __init__(self) -> None:
        """
        Docker container.
        """
        raise NotImplemented
    
    async def _run_docker_compose(self) -> None:
        """
        Run docker container initialization script.
        """
        raise NotImplemented
    
@dataclass
class MCPServerConfig:
    """
    Config for the MCP server that the runtime will use.
    """
    raise NotImplemented
    