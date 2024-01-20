# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
from pyrit.chat.azure_openai_chat import AzureOpenAIChat

from pyrit.prompt_target.prompt_target import PromptTarget


class AzureOpenAIChatTartget(PromptTarget, AzureOpenAIChat):
    def __init__(
        self,
        *,
        deployment_name: str,
        endpoint: str,
        api_key: str,
        api_version: str = "2023-08-01-preview",
    ) -> None:
        super().__init__(deployment_name=deployment_name, endpoint=endpoint, api_key=api_key, api_version=api_version)

    def send_prompt(self, normalized_prompt: str) -> None:
        return super().send_prompt(normalized_prompt)
