# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from openai import AzureOpenAI

from pyrit.common import default_values
from pyrit.interfaces import CompletionSupport
from pyrit.models import PromptResponse


class AzureCompletion(CompletionSupport):
    API_KEY_ENVIRONMENT_VARIABLE: str = "AZURE_OPENAI_COMPLETION_KEY"
    ENDPOINT_URI_ENVIRONMENT_VARIABLE: str = "AZURE_OPENAI_COMPLETION_ENDPOINT"
    DEPLOYMENT_ENVIRONMENT_VARIABLE: str = "AZURE_OPENAI_COMPLETION_DEPLOYMENT"

    def __init__(
        self, api_key: str = None, endpoint: str = None, deployment: str = None, api_version: str = "2023-05-15"
    ):
        """
        Initializes an instance of the AzureCompletions class.

        Args:
            api_key (str, optional): The API key for accessing the Azure OpenAI service.
                                     Defaults to environment variable.
            endpoint (str, optional): The endpoint URL for the Azure OpenAI service.
                                      Defaults to environment variable.
            deployment (str, optional): The deployment name for the Azure OpenAI service.
                                        Defaults to environment variable.
            api_version (str, optional): The API version for the Azure OpenAI service. Defaults to "2023-05-15".
        """

        api_key = default_values.get_required_value(
            env_var_name=self.API_KEY_ENVIRONMENT_VARIABLE, passed_value=api_key
        )
        endpoint = default_values.get_required_value(
            env_var_name=self.ENDPOINT_URI_ENVIRONMENT_VARIABLE, passed_value=endpoint
        )
        self._model = default_values.get_required_value(
            env_var_name=self.DEPLOYMENT_ENVIRONMENT_VARIABLE, passed_value=deployment
        )

        self._client = AzureOpenAI(
            api_key=api_key,
            api_version=api_version,
            azure_endpoint=endpoint,
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
