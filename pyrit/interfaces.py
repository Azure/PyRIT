# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

import abc
from abc import abstractmethod
from dataclasses import dataclass

from pyrit.models import EmbeddingResponse, PromptResponse


class Authenticator(abc.ABC):
    token: str

    @abstractmethod
    def refresh_token(self) -> str:
        raise NotImplementedError("refresh_token method not implemented")

    @abstractmethod
    def get_token(self) -> str:
        raise NotImplementedError("get_token method not implemented")


@dataclass
class LLMEndpoint(abc.ABC):
    @abstractmethod
    def rotate_api_key(self, token: str) -> None:
        """
        Update the API key used in calls
        Args:
            token: The API key to be used

        Returns:
            None:
        """
        raise NotImplementedError("rotate_api_key method not implemented")

    @abstractmethod
    def complete_text(self, prompt: str, temperature: float = 0.7, max_tokens: int = 1_024) -> PromptResponse:
        """
        Autocomplete a prompt.

        Args:
            max_tokens:
            temperature:
            prompt: User prompt

        Returns:
            Text response given by the model
        """
        raise NotImplementedError("complete_text method not implemented")

    @abstractmethod
    def text_to_embedding(self, prompt: str) -> list[float]:
        """
        Convert a text prompt to an embedding

        Args:
            prompt: User prompt

        Returns:
            A list of floats corresponding to the text's embedding.
        """
        raise NotImplementedError("text_to_embedding method not implemented")


class EmbeddingSupport(abc.ABC):
    @abstractmethod
    def generate_text_embedding(self, text: str, **kwargs) -> EmbeddingResponse:
        """Generate text embedding

        Args:
            text: The text to generate the embedding for
            **kwargs: Additional arguments to pass to the function.

        Returns:
            The embedding response
        """
        raise NotImplementedError("generate_text_embedding method not implemented")


class CompletionSupport(abc.ABC):
    @abstractmethod
    def complete_text(self, text: str, **kwargs) -> PromptResponse:
        """Complete text based on a given prompt

        Args:
            text:  The prompt to complete

        Returns:
            The completed text
        """
        raise NotImplementedError("complete_text method not implemented")

    @abstractmethod
    async def complete_text_async(self, text: str, **kwargs) -> PromptResponse:
        """Complete text based on a given prompt

        Args:
            text:  The prompt to complete

        Returns:
            The completed text
        """
        raise NotImplementedError("complete_text_async method not implemented")
