# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import json
import logging
import re
from abc import abstractmethod
from typing import Optional
from urllib.parse import urlparse

from pyrit.auth import AzureAuth
from pyrit.auth.azure_auth import get_default_scope
from pyrit.common import default_values
from pyrit.prompt_target import PromptChatTarget

logger = logging.getLogger(__name__)


class OpenAITarget(PromptChatTarget):

    ADDITIONAL_REQUEST_HEADERS: str = "OPENAI_ADDITIONAL_REQUEST_HEADERS"

    model_name_environment_variable: str
    endpoint_environment_variable: str
    api_key_environment_variable: str

    _azure_auth: Optional[AzureAuth] = None

    def __init__(
        self,
        *,
        model_name: Optional[str] = None,
        endpoint: Optional[str] = None,
        api_key: Optional[str] = None,
        headers: Optional[str] = None,
        use_entra_auth: bool = False,
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
            api_key (str, Optional): The API key for accessing the Azure OpenAI service (only if you're not using
                Entra authentication). Defaults to the `OPENAI_CHAT_KEY` environment variable.
            headers (str, Optional): Extra headers of the endpoint (JSON).
            use_entra_auth (bool): When set to True, user authentication is used
                instead of API Key. DefaultAzureCredential is taken for
                https://cognitiveservices.azure.com/.default. Please run `az login` locally
                to leverage user AuthN.
            max_requests_per_minute (int, Optional): Number of requests the target can handle per
                minute before hitting a rate limit. The number of requests sent to the target
                will be capped at the value provided.
            httpx_client_kwargs (dict, Optional): Additional kwargs to be passed to the
                `httpx.AsyncClient()` constructor.
        """
        self._headers: dict = {}
        self._httpx_client_kwargs = httpx_client_kwargs or {}

        request_headers = default_values.get_non_required_value(
            env_var_name=self.ADDITIONAL_REQUEST_HEADERS, passed_value=headers
        )

        if request_headers and isinstance(request_headers, str):
            self._headers = json.loads(request_headers)

        self._set_openai_env_configuration_vars()

        self._model_name: str = default_values.get_non_required_value(
            env_var_name=self.model_name_environment_variable, passed_value=model_name
        )
        endpoint_value = default_values.get_required_value(
            env_var_name=self.endpoint_environment_variable, passed_value=endpoint
        )

        # Initialize parent with endpoint and model_name
        PromptChatTarget.__init__(
            self, max_requests_per_minute=max_requests_per_minute, endpoint=endpoint_value, model_name=self._model_name
        )

        self._api_key = api_key

        self._set_auth_headers(use_entra_auth=use_entra_auth, passed_api_key=api_key)

    def _set_auth_headers(self, use_entra_auth, passed_api_key) -> None:
        if use_entra_auth:
            if passed_api_key:
                raise ValueError("If using Entra ID auth, please do not specify api_key.")
            logger.info("Authenticating with AzureAuth")
            scope = get_default_scope(self._endpoint)
            self._azure_auth = AzureAuth(token_scope=scope)
            self._headers["Authorization"] = f"Bearer {self._azure_auth.get_token()}"
            self._api_key = None
        else:
            self._api_key = default_values.get_non_required_value(
                env_var_name=self.api_key_environment_variable, passed_value=passed_api_key
            )
            # This header is set as api-key in azure and bearer in openai
            # But azure still functions if it's in both places and in fact,
            # in Azure foundry it needs to be set as a bearer
            self._headers["Api-Key"] = self._api_key
            self._headers["Authorization"] = f"Bearer {self._api_key}"

    def refresh_auth_headers(self) -> None:
        """
        Refresh the authentication headers. This is particularly useful for Entra authentication
        where tokens need to be refreshed periodically.
        """
        if self._azure_auth:
            self._headers["Authorization"] = f"Bearer {self._azure_auth.refresh_token()}"

    @abstractmethod
    def _set_openai_env_configuration_vars(self) -> None:
        """
        Sets deployment_environment_variable, endpoint_environment_variable, and api_key_environment_variable
        which are read from .env
        """  # noqa: D415
        raise NotImplementedError

    def _warn_if_irregular_endpoint(self, expected_url_regex) -> None:
        """
        Validate that the endpoint URL ends with one of the expected routes for this OpenAI target.

        Args:
            expected_url_regex: Expected regex pattern(s) for this target. Should be a list of regex strings.

        Prints a warning if the endpoint doesn't match any of the expected routes.
        This validation helps ensure the endpoint is configured correctly for the specific API.
        """
        if not self._endpoint or not expected_url_regex:
            return

        # Use urllib to extract the path part and normalize it
        parsed_url = urlparse(self._endpoint)
        normalized_route = parsed_url.path.lower().rstrip("/")

        # Check if the endpoint matches any of the expected regex patterns
        for regex_pattern in expected_url_regex:
            if re.search(regex_pattern, normalized_route):
                return

        # No matches found, log warning
        if len(expected_url_regex) == 1:
            # Convert regex back to human-readable format for the warning
            pattern_str = expected_url_regex[0].replace(r"[^/]+", "*").replace("$", "")
            expected_routes_str = pattern_str
        else:
            # Convert all regex patterns to human-readable format
            readable_patterns = [p.replace(r"[^/]+", "*").replace("$", "") for p in expected_url_regex]
            expected_routes_str = f"one of: {', '.join(readable_patterns)}"

        logger.warning(
            f"The provided endpoint URL {parsed_url} does not match any of the expected formats: {expected_routes_str}."
            f"This may be intentional, especially if you are using an endpoint other than Azure or OpenAI."
            f"For more details and guidance, please see the .env_example file in the repository."
        )

    @abstractmethod
    def is_json_response_supported(self) -> bool:
        """
        Abstract method to determine if JSON response format is supported by the target.

        Returns:
            bool: True if JSON response is supported, False otherwise.
        """
        pass
