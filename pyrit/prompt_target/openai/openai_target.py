# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import json
import logging
from abc import abstractmethod
from typing import Optional

from openai import AsyncAzureOpenAI, AsyncOpenAI

from pyrit.auth.azure_auth import (
    get_token_provider_from_default_azure_credential,
    get_default_scope
)
from pyrit.common import default_values
from pyrit.prompt_target import PromptChatTarget

logger = logging.getLogger(__name__)


class OpenAITarget(PromptChatTarget):

    ADDITIONAL_REQUEST_HEADERS: str = "OPENAI_ADDITIONAL_REQUEST_HEADERS"

    model_name_environment_variable: str
    target_uri_environment_variable: str
    api_key_environment_variable: str

    def __init__(
        self,
        *,
        model_name: str = None,
        target_uri: str = None,
        api_key: str = None,
        headers: str = None,
        use_aad_auth: bool = False,
        api_version: str = "2024-06-01",
        max_requests_per_minute: Optional[int] = None,
    ) -> None:
        """
        Abstract class that initializes an Azure or non-Azure OpenAI chat target.

        Read more about the various models here:
        https://learn.microsoft.com/en-us/azure/ai-services/openai/concepts/models.


        Args:
            model_name (str, Optional): The name of the model.
            target_uri (str, Optional): The target URL for the OpenAI service.
            api_key (str, Optional): The API key for accessing the Azure OpenAI service.
                Defaults to the AZURE_OPENAI_CHAT_KEY environment variable.
            headers (str, Optional): Headers of the endpoint (JSON).
            use_aad_auth (bool, Optional): When set to True, user authentication is used
                instead of API Key. DefaultAzureCredential is taken for
                https://cognitiveservices.azure.com/.default . Please run `az login` locally
                to leverage user AuthN.
            api_version (str, Optional): The version of the Azure OpenAI API. Defaults to
                "2024-06-01".
            max_requests_per_minute (int, Optional): Number of requests the target can handle per
                minute before hitting a rate limit. The number of requests sent to the target
                will be capped at the value provided.
        """
        PromptChatTarget.__init__(self, max_requests_per_minute=max_requests_per_minute)

        self._extra_headers: dict = {}

        request_headers = default_values.get_non_required_value(
            env_var_name=self.ADDITIONAL_REQUEST_HEADERS, passed_value=headers
        )

        if request_headers and isinstance(request_headers, str):
            self._extra_headers = json.loads(request_headers)

        self._api_version = api_version

        self._set_openai_env_configuration_vars()

        self._model_name = default_values.get_non_required_value(
            env_var_name=self.model_name_environment_variable, passed_value=model_name
        )
        self._target_uri = default_values.get_required_value(
            env_var_name=self.target_uri_environment_variable, passed_value=target_uri
        ).rstrip("/")


        self._api_key = None
        self._token_provider = None

        if use_aad_auth:
            logger.info("Authenticating with DefaultAzureCredential() for Azure Cognitive Services")

            scope = get_default_scope(self._target_uri)
            self._token_provider = get_token_provider_from_default_azure_credential(scope=scope)

        else:
            self._api_key = default_values.get_required_value(
                env_var_name=self.api_key_environment_variable, passed_value=api_key
            )

    @abstractmethod
    def _set_openai_env_configuration_vars(self) -> None:
        """
        Sets deployment_environment_variable, target_uri_environment_variable, and api_key_environment_variable
        which are read from .env
        """
        raise NotImplementedError

    @abstractmethod
    def is_json_response_supported(self) -> bool:
        """
        Abstract method to determine if JSON response format is supported by the target.

        Returns:
            bool: True if JSON response is supported, False otherwise.
        """
        pass
