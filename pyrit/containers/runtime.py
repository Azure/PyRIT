# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

import functools
import logging
from typing import Any, Callable, Dict, TypeVar, Optional
from pyrit.models.message import Message
from pyrit.models.message_piece import MessagePiece
from pyrit.containers.utils import DockerComposeConfig, DockerImageFile
import uuid

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
        environment, e.g. a Docker container. Timeline:
        1. PromptTarget is created with non-None runtime.
        2. PromptTarget.send_prompt_async is called for request.
        3. Runtime.send_async is called.
        
        Note that this only supports sequential tool execution (assistant -> system -> memory)
        not (assistant <-loop-> system) --> memory.
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
            Message (original request with server response.)
        """
        
        mcp_request = self._encode_mcp(request)
        # TODO: This method doesn't exist. Need to use fastmcp + httpx to make HTTPS request
        client_response = self._client.send()
        new_piece = self._decode_mcp(client_response)
        old_pieces = list(request.message_pieces)
        response = Message(
            message_pieces=old_pieces + [new_piece]
        )
        return response
    
    @classmethod
    def _encode_mcp(cls, contents: Message) -> Dict:
        """
        Encode Message object into HTTPS request for MCP server.
        Use fastmcp + httpx.
        """
        raise NotImplemented
    
    @classmethod
    def _decode_mcp(cls, contents: Dict) -> MessagePiece:
        """
        Decode HTTPS response from MCP into MessagePiece.
        Assign role "system" for container outputs.
        """
        raise NotImplemented