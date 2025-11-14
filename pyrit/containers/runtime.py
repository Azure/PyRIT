# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

import functools
import logging
from typing import Any, Callable, TypeVar, Optional
from pyrit.models.message import Message
from pyrit.models.message_piece import MessagePiece
from pyrit.containers.utils import DockerComposeConfig, DockerImageFile

import docker

logger = logging.getLogger(__name__)

def wrap_runtime(func: Callable) -> Callable:
    """
    Decorator that wraps send_prompt_async to integrate runtime processing.
    
    Flow:
    1. Call the original send_prompt_async to get the target's response.
    2. If runtime is configured, feed the response to runtime.send_async.
    3. Return the runtime-processed result.
    
    Args:
        func: The async method to wrap (here, PromptTarget.send_prompt_async).
        
    Returns:
        Callable: The wrapped async method.
    """
    @functools.wraps(func)
    async def wrapper(self, *, message: Message) -> Message:
        response = await func(self, message=message)

        if self._runtime:
            logger.info(f"Processing response through runtime: {self._runtime}")
            runtime_response = await self._runtime.send_async(response)
            return runtime_response

        # If there is no runtime, silently return the model output.
        return response
    
    return wrapper


class Runtime:
    def __init__(self, *, image: DockerImageFile, config: DockerComposeConfig) -> None:
        """
        The runtime object enables a PromptTarget to access a simulated
        environment, e.g. a Docker container.
        >>> PromptTarget._runtime = <runtime_object at xxxx>
        >>> PromptTarget.send_prompt_async calls PromptTarget._runtime.send_async to parse messages.
        """

        self._client = docker.from_env()
        self._client.containers.run(
            image=str(image),
            **config.unpack()
        )
        
    async def send_async(self, request: Message) -> Message:
        """
        Process the target input (called from PromptTarget) into a result.
        Returns: 
        """
        contents = [c.original_value + "\n" for c in request.message_pieces]
        output = "Not Implemented"
        # TODO: output = self._client.mcp.send(contents)
        new_piece = MessagePiece(role="system", original_value=output)
        response = Message(
            message_pieces=request.message_pieces
        )

