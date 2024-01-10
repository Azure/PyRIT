# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from openai import AzureOpenAI

from pyrit.interfaces import CompletionSupport
from pyrit.models import PromptResponse


class AzureCompletion(CompletionSupport):
    def __init__(self, api_key: str, api_base: str, model: str, api_version: str = "2023-05-15"):
        self._model = model
        self._api_version = api_version
        self._api_base = api_base

        self._client = AzureOpenAI(
            api_key=api_key,
            api_version=api_version,
            azure_endpoint=api_base,
        )

    def complete_text(self, text: str, **kwargs) -> PromptResponse:
        """Complete the text using the Azure Completion API.
        Args:
            text: The text to complete.
            **kwargs: Additional keyword arguments to pass to the Azure Completion API.
        Returns:
            A PromptResponse object with the completion response from the Azure Completion API.
        """
        response = self._client.completions.create(model=self._model, prompt=text, **kwargs)
        # return response
        prompt_response = PromptResponse(
            completion=response.choices[0].text,
            prompt=text,
            id=response.id,
            completion_tokens=response.usage.completion_tokens,
            prompt_tokens=response.usage.prompt_tokens,
            total_tokens=response.usage.total_tokens,
            model=response.model,
            object=response.object,
        )
        return prompt_response

    async def complete_text_async(self, text: str, **kwargs) -> PromptResponse:
        return await super().complete_text_async(text=text, **kwargs)
