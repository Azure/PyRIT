# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import json
import logging
from abc import abstractmethod
from typing import List, Optional, Union
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
    _expected_route: Optional[Union[str, List[str]]] = None

    def __init__(
        self,
        *,
        model_name: Optional[str] = None,
        endpoint: Optional[str] = None,
        api_key: Optional[str] = None,
        headers: Optional[str] = None,
        use_entra_auth: bool = False,
        api_version: Optional[str] = "2024-10-21",
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
            api_version (str, Optional): The version of the Azure OpenAI API. Defaults to
                "2024-06-01". If set to None, this will not be added as a query parameter to requests.
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

        self._api_version = api_version

        self._set_openai_env_configuration_vars()

        self._model_name: str = default_values.get_non_required_value(
            env_var_name=self.model_name_environment_variable, passed_value=model_name
        )
        endpoint_value = default_values.get_required_value(
            env_var_name=self.endpoint_environment_variable, passed_value=endpoint
        ).rstrip("/")

        # Initialize parent with endpoint and model_name
        PromptChatTarget.__init__(
            self, max_requests_per_minute=max_requests_per_minute, endpoint=endpoint_value, model_name=self._model_name
        )

        self._api_key = api_key

        self._set_auth_headers(use_entra_auth=use_entra_auth, passed_api_key=api_key)

        # Validate endpoint URL format
        self._warn_if_irregular_endpoint()

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
        """Refresh the authentication headers. This is particularly useful for Entra authentication
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

    def _warn_if_irregular_endpoint(self) -> None:
        """
        Validate that the endpoint URL ends with one of the expected routes for this OpenAI target.

        Prints a warning if the endpoint doesn't match any of the expected routes.
        This validation helps ensure the endpoint is configured correctly for the specific API.
        """
        if not self._endpoint or not self._expected_route:
            return

        # Use urllib to extract the path part and normalize it
        parsed_url = urlparse(self._endpoint)
        normalized_route = parsed_url.path.lower().rstrip("/")

        # Handle both single route (string) and multiple routes (list)
        expected_routes = self._expected_route if isinstance(self._expected_route, list) else [self._expected_route]

        # Check if the endpoint matches any of the expected routes
        for expected_route in expected_routes:
            if expected_route is None:
                continue

            expected_route = expected_route.lower().rstrip("/")

            # Handle wildcard patterns like "/openai/deployments/*/chat/completions"
            if "*" in expected_route:
                if self._matches_wildcard_pattern(normalized_route, expected_route):
                    return
            else:
                # Exact matching for routes without wildcards
                if normalized_route == expected_route:
                    return

        # No matches found, log warning
        expected_routes_str = (
            str(self._expected_route)
            if isinstance(self._expected_route, str)
            else f"one of: {', '.join(self._expected_route)}"
        )
        logger.warning(
            f"Expected endpoint to end with  {expected_routes_str} "
            f"Please verify your endpoint URL: '{self._endpoint}'."
        )

    def _matches_wildcard_pattern(self, route: str, pattern: str) -> bool:
        """Check if a route matches a wildcard pattern."""
        pattern_parts = pattern.split("/")
        route_parts = route.split("/")

        if len(pattern_parts) != len(route_parts):
            return False

        for pattern_part, route_part in zip(pattern_parts, route_parts):
            if pattern_part != "*" and pattern_part != route_part:
                return False

        return True

    @abstractmethod
    def is_json_response_supported(self) -> bool:
        """
        Abstract method to determine if JSON response format is supported by the target.

        Returns:
            bool: True if JSON response is supported, False otherwise.
        """
        pass
