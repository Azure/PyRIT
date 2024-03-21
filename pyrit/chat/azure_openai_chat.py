# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
from openai import AsyncAzureOpenAI, AzureOpenAI
from openai.types.chat import ChatCompletion
from pyrit.common import default_values

from pyrit.interfaces import ChatSupport
from pyrit.models import ChatMessage

from pyrit.chat.openai_base import OpenAIBase

class AzureOpenAIChat(OpenAIBase):
    DEPLOYMENT_ENVIRONMENT_VARIABLE: str = "AZURE_OPENAI_CHAT_DEPLOYMENT"

    def __init__(
        self,
        *,
        deployment_name: str = None,
        endpoint: str = None,
        api_key: str = None,
        api_version: str = "2023-08-01-preview",
    ) -> None:
        self._deployment_name = default_values.get_required_value(
            env_var_name=self.DEPLOYMENT_ENVIRONMENT_VARIABLE, passed_value=deployment_name
        )

        endpoint = default_values.get_required_value(
            env_var_name=super().ENDPOINT_URI_ENVIRONMENT_VARIABLE, passed_value=endpoint
        )
        api_key = default_values.get_required_value(
            env_var_name=super().API_KEY_ENVIRONMENT_VARIABLE, passed_value=api_key
        )

        _client = AzureOpenAI(
            api_key=api_key,
            api_version=api_version,
            azure_endpoint=endpoint,
        )
        _async_client = AsyncAzureOpenAI(
            api_key=api_key,
            api_version=api_version,
            azure_endpoint=endpoint,
        )
        super().__init__(client=_client, async_client=_async_client) 
