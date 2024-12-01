# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import json
import logging
from abc import abstractmethod
from typing import Optional

from openai import AsyncAzureOpenAI, AsyncOpenAI

from pyrit.auth.azure_auth import get_token_provider_from_default_azure_credential
from pyrit.common import default_values
from pyrit.prompt_target import PromptChatTarget

logger = logging.getLogger(__name__)


class OpenAITarget(PromptChatTarget):

    ADDITIONAL_REQUEST_HEADERS: str = "OPENAI_ADDITIONAL_REQUEST_HEADERS"

    deployment_environment_variable: str
    endpoint_uri_environment_variable: str
    api_key_environment_variable: str

    def __init__(
        self,
        *,
        deployment_name: str = None,
        endpoint: str = None,
        api_key: str = None,
        headers: str = None,
        is_azure_target=True,
        use_aad_auth: bool = False,
        api_version: str = "2024-06-01",
        max_requests_per_minute: Optional[int] = None,
    ) -> None:
        """
        Abstract class that initializes an Azure or non-Azure OpenAI chat target.

        Read more about the various models here:
        https://learn.microsoft.com/en-us/azure/ai-services/openai/concepts/models.


        Args:
            deployment_name (str, Optional): The name of the deployment. Defaults to the
                AZURE_OPENAI_CHAT_DEPLOYMENT environment variable .
            endpoint (str, Optional): The endpoint URL for the Azure OpenAI service.
                Defaults to the AZURE_OPENAI_CHAT_ENDPOINT environment variable.
            api_key (str, Optional): The API key for accessing the Azure OpenAI service.
                Defaults to the AZURE_OPENAI_CHAT_KEY environment variable.
            headers (str, Optional): Headers of the endpoint (JSON).
            is_azure_target (bool, Optional): Whether the target is an Azure target.
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
        self._is_azure_target = is_azure_target

        if self._is_azure_target:
            self._initialize_azure_vars(deployment_name, endpoint, api_key, use_aad_auth)
        else:
            # Initialize for non-Azure OpenAI
            self._initialize_non_azure_vars(deployment_name, endpoint, api_key)
            if not self._deployment_name:
                # OpenAI deployments listed here: https://platform.openai.com/docs/models
                raise ValueError("The deployment name must be provided for non-Azure OpenAI targets. e.g. gpt-4o")

    def _initialize_azure_vars(self, deployment_name: str, endpoint: str, api_key: str, use_aad_auth: bool):
        self._set_azure_openai_env_configuration_vars()

        self._deployment_name = default_values.get_required_value(
            env_var_name=self.deployment_environment_variable, passed_value=deployment_name
        )
        self._endpoint = default_values.get_required_value(
            env_var_name=self.endpoint_uri_environment_variable, passed_value=endpoint
        ).rstrip("/")

        if use_aad_auth:
            logger.info("Authenticating with DefaultAzureCredential() for Azure Cognitive Services")
            token_provider = get_token_provider_from_default_azure_credential()

            self._async_client = AsyncAzureOpenAI(
                azure_ad_token_provider=token_provider,
                api_version=self._api_version,
                azure_endpoint=self._endpoint,
                default_headers=self._extra_headers,
            )
        else:
            self._api_key = default_values.get_required_value(
                env_var_name=self.api_key_environment_variable, passed_value=api_key
            )

            self._async_client = AsyncAzureOpenAI(
                api_key=self._api_key,
                api_version=self._api_version,
                azure_endpoint=self._endpoint,
                default_headers=self._extra_headers,
            )

    def _initialize_non_azure_vars(self, deployment_name: str, endpoint: str, api_key: str):
        """
        Initializes variables to communicate with the (non-Azure) OpenAI API
        """
        self._api_key = default_values.get_required_value(env_var_name="OPENAI_KEY", passed_value=api_key)
        if not self._api_key:
            raise ValueError("API key for OpenAI is missing. Ensure OPENAI_KEY is set in the environment.")

        # Any available model. See https://platform.openai.com/docs/models
        self._deployment_name = default_values.get_required_value(
            env_var_name="OPENAI_DEPLOYMENT", passed_value=deployment_name
        )
        if not self._deployment_name:
            raise ValueError(
                "Deployment name for OpenAI is missing. Ensure OPENAI_DEPLOYMENT is set in the environment."
            )

        endpoint = endpoint if endpoint else "https://api.openai.com/v1/chat/completions"

        # Ignoring mypy type error. The OpenAI client and Azure OpenAI client have the same private base class
        self._async_client = AsyncOpenAI(  # type: ignore
            api_key=self._api_key,
            default_headers=self._extra_headers,
        )

    @abstractmethod
    def _set_azure_openai_env_configuration_vars(self) -> None:
        """
        Sets deployment_environment_variable, endpoint_uri_environment_variable, and api_key_environment_variable
        which are read from .env
        """
        raise NotImplementedError
