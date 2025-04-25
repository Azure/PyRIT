# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import json
import logging
from abc import abstractmethod
from typing import Optional

from pyrit.auth.azure_auth import (
    AzureAuth,
    get_default_scope,
)
from pyrit.common import default_values
from pyrit.prompt_target import PromptChatTarget

logger = logging.getLogger(__name__)


class OpenAITarget(PromptChatTarget):

    ADDITIONAL_REQUEST_HEADERS: str = "OPENAI_ADDITIONAL_REQUEST_HEADERS"

    model_name_environment_variable: str
    endpoint_environment_variable: str
    api_key_environment_variable: str

    _model_name: Optional[str]
    _azure_auth: Optional[AzureAuth] = None

    def __init__(
        self,
        *,
        model_name: Optional[str] = None,
        endpoint: Optional[str] = None,
        api_key: Optional[str] = None,
        headers: Optional[str] = None,
        use_aad_auth: Optional[bool] = False,
        api_version: Optional[str] = "2024-06-01",
        max_requests_per_minute: Optional[int] = None,
        httpx_client_kwargs: Optional[dict] = None,
    ) -> None:
        """
        Abstract class that initializes an Azure or non-Azure OpenAI chat target.

        Read more about the various models here:
        https://learn.microsoft.com/en-us/azure/ai-services/openai/concepts/models.


        Args:
            model_name (str, Optional): The name of the model.
            endpoint (str, Optional): The target URL for the OpenAI service.
            api_key (str, Optional): The API key for accessing the Azure OpenAI service.
                Defaults to the OPENAI_CHAT_KEY environment variable.
            headers (str, Optional): Extra headers of the endpoint (JSON).
            use_aad_auth (bool, Optional): When set to True, user authentication is used
                instead of API Key. DefaultAzureCredential is taken for
                https://cognitiveservices.azure.com/.default . Please run `az login` locally
                to leverage user AuthN.
            api_version (str, Optional): The version of the Azure OpenAI API. Defaults to
                "2024-06-01". If set to None, this will not be added as a query parameter to requests.
            max_requests_per_minute (int, Optional): Number of requests the target can handle per
                minute before hitting a rate limit. The number of requests sent to the target
                will be capped at the value provided.
            httpx_client_kwargs (dict, Optional): Additional kwargs to be passed to the
                httpx.AsyncClient() constructor.
        """
        PromptChatTarget.__init__(self, max_requests_per_minute=max_requests_per_minute)

        self._headers: dict = {}
        self._httpx_client_kwargs = httpx_client_kwargs or {}

        request_headers = default_values.get_non_required_value(
            env_var_name=self.ADDITIONAL_REQUEST_HEADERS, passed_value=headers
        )

        if request_headers and isinstance(request_headers, str):
            self._headers = json.loads(request_headers)

        self._api_version = api_version

        self._set_openai_env_configuration_vars()

        self._model_name: str = default_values.get_non_required_value(
            env_var_name=self.model_name_environment_variable, passed_value=model_name
        )
        self._endpoint = default_values.get_required_value(
            env_var_name=self.endpoint_environment_variable, passed_value=endpoint
        ).rstrip("/")

        self._api_key = api_key

        self._set_auth_headers(use_aad_auth=use_aad_auth, passed_api_key=api_key)

    def _set_auth_headers(self, use_aad_auth, passed_api_key) -> None:
        self._api_key = default_values.get_non_required_value(
            env_var_name=self.api_key_environment_variable, passed_value=passed_api_key
        )

        if self._api_key:
            # This header is set as api-key in azure and bearer in openai
            # But azure still functions if it's in both places and in fact,
            # in Azure foundry it needs to be set as a bearer
            self._headers["Api-Key"] = self._api_key
            self._headers["Authorization"] = f"Bearer {self._api_key}"

        if use_aad_auth:
            logger.info("Authenticating with AzureAuth")
            scope = get_default_scope(self._endpoint)
            self._azure_auth = AzureAuth(token_scope=scope)
            self._headers["Authorization"] = f"Bearer {self._azure_auth.get_token()}"

    def refresh_auth_headers(self) -> None:
        """Refresh the authentication headers. This is particularly useful for AAD authentication
        where tokens need to be refreshed periodically."""
        if self._azure_auth:
            self._headers["Authorization"] = f"Bearer {self._azure_auth.refresh_token()}"

    @abstractmethod
    def _set_openai_env_configuration_vars(self) -> None:
        """
        Sets deployment_environment_variable, endpoint_environment_variable, and api_key_environment_variable
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
