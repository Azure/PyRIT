# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import json
import logging
import re
from abc import abstractmethod
from typing import Optional, Union
from urllib.parse import urlparse

from openai import AsyncOpenAI, AsyncAzureOpenAI

from pyrit.auth import AzureAuth
from pyrit.auth.azure_auth import get_async_token_provider_from_default_azure_credential, get_default_scope
from pyrit.common import default_values
from pyrit.prompt_target import PromptChatTarget

logger = logging.getLogger(__name__)


class OpenAITarget(PromptChatTarget):

    ADDITIONAL_REQUEST_HEADERS: str = "OPENAI_ADDITIONAL_REQUEST_HEADERS"

    model_name_environment_variable: str
    endpoint_environment_variable: str
    api_key_environment_variable: str

    _azure_auth: Optional[AzureAuth] = None
    _async_client: Optional[Union[AsyncOpenAI, AsyncAzureOpenAI]] = None

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
        self._use_entra_auth = use_entra_auth

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
        
        # For Azure endpoints with deployment in URL, extract it if model_name not provided
        if not self._model_name and "azure" in endpoint_value.lower():
            extracted = self._extract_deployment_from_azure_url(endpoint_value)
            if extracted:  # Only use extracted deployment if we actually found one
                self._model_name = extracted

        # Initialize parent with endpoint and model_name
        PromptChatTarget.__init__(
            self, max_requests_per_minute=max_requests_per_minute, endpoint=endpoint_value, model_name=self._model_name
        )

        self._api_key = api_key

        self._set_auth_headers(use_entra_auth=use_entra_auth, passed_api_key=api_key)
        self._initialize_openai_client()

    def _set_auth_headers(self, use_entra_auth, passed_api_key) -> None:
        if use_entra_auth:
            if passed_api_key:
                raise ValueError("If using Entra ID auth, please do not specify api_key.")
            logger.info("Authenticating with AzureAuth")
            scope = get_default_scope(self._endpoint)
            self._azure_auth = AzureAuth(token_scope=scope)
            # For SDK-based targets: auth is handled via azure_ad_token_provider parameter
            # For non-SDK targets (DALL-E, TTS, etc): keep manual header for backward compatibility
            self._headers["Authorization"] = f"Bearer {self._azure_auth.get_token()}"
            self._api_key = None
        else:
            self._api_key = default_values.get_non_required_value(
                env_var_name=self.api_key_environment_variable, passed_value=passed_api_key
            )
            # For SDK-based targets: api_key is passed to AsyncOpenAI/AsyncAzureOpenAI constructors
            # For non-SDK targets (DALL-E, TTS, etc): keep manual headers for backward compatibility
            self._headers["Api-Key"] = self._api_key
            self._headers["Authorization"] = f"Bearer {self._api_key}"

    def _extract_deployment_from_azure_url(self, url: str) -> str:
        """
        Extract deployment/model name from Azure OpenAI URL.
        
        Azure URLs have formats like:
        - https://{resource}.openai.azure.com/openai/deployments/{deployment}/chat/completions
        - https://{resource}.openai.azure.com/openai/deployments/{deployment}/responses
        
        Args:
            url: The Azure endpoint URL.
            
        Returns:
            The deployment name, or empty string if not found.
        """
        # Match /deployments/{deployment_name}/
        match = re.search(r'/deployments/([^/]+)/', url)
        if match:
            deployment = match.group(1)
            logger.info(f"Extracted deployment name from URL: {deployment}")
            return deployment
        
        return ""

    def refresh_auth_headers(self) -> None:
        """
        Refresh the authentication headers. This is particularly useful for Entra authentication
        where tokens need to be refreshed periodically.
        """
        if self._azure_auth:
            self._headers["Authorization"] = f"Bearer {self._azure_auth.refresh_token()}"

    def _initialize_openai_client(self) -> None:
        """
        Initialize the OpenAI client based on whether it's Azure or standard OpenAI.
        
        Azure has multiple endpoint formats:
        1. Old format: https://{resource}.openai.azure.com/openai/deployments/{deployment}/...?api-version=...
           Uses AsyncAzureOpenAI client with api-version parameter (supports both API key and Entra auth)
        2. New format: https://{resource}.openai.azure.com/openai/v1 OR https://{resource}.models.ai.azure.com/...
           Uses standard AsyncOpenAI client (no api-version needed)
           - With API key: pass api_key parameter
           - With Entra: pass token provider function as api_key parameter
        """
        # Determine if this is Azure OpenAI based on the endpoint
        is_azure = "azure" in self._endpoint.lower() if self._endpoint else False
        
        # Check if it's the new Azure format (OpenAI-compatible)
        # New format includes:
        # - https://{resource}.openai.azure.com/openai/v1
        # - https://{resource}.models.ai.azure.com/... (Azure Foundry endpoints)
        is_azure_new_format = False
        if is_azure:
            from urllib.parse import urlparse
            
            parsed_url = urlparse(self._endpoint)
            # New format has /openai/v1 path OR uses models.ai.azure.com domain
            is_azure_new_format = "/openai/v1" in parsed_url.path or ".models.ai.azure.com" in parsed_url.netloc
        
        # Merge custom headers with httpx_client_kwargs
        httpx_kwargs = self._httpx_client_kwargs.copy()
        if self._headers:
            httpx_kwargs.setdefault("default_headers", {}).update(self._headers)
        
        # Only old Azure format uses AsyncAzureOpenAI
        # Everything else (platform OpenAI, new Azure format, Azure Foundry) uses standard AsyncOpenAI
        if is_azure and not is_azure_new_format:
            # Old Azure format - uses AsyncAzureOpenAI client
            # Endpoint format: https://{resource}.openai.azure.com/openai/deployments/{deployment}/...
            
            # Extract API version from query parameter if present
            import os
            from urllib.parse import parse_qs
            
            parsed_url = urlparse(self._endpoint)
            query_params = parse_qs(parsed_url.query)
            
            # Get api_version from query param, environment variable, or default
            if "api-version" in query_params:
                api_version = query_params["api-version"][0]
            else:
                api_version = os.environ.get("OPENAI_API_VERSION", "2024-02-15-preview")
            
            # Azure SDK expects ONLY the base endpoint (scheme://netloc)
            # It will automatically append paths like /openai/deployments/{deployment}/chat/completions
            azure_endpoint = f"{parsed_url.scheme}://{parsed_url.netloc}"
            
            # Get the token provider for Entra auth
            azure_ad_token_provider = None
            if self._use_entra_auth and self._azure_auth:
                # Create a token provider function for async operations
                async def token_provider():
                    return self._azure_auth.refresh_token()
                azure_ad_token_provider = token_provider
            
            self._async_client = AsyncAzureOpenAI(
                azure_endpoint=azure_endpoint,
                api_version=api_version,
                api_key=self._api_key if not self._use_entra_auth else None,
                azure_ad_token_provider=azure_ad_token_provider,
                **httpx_kwargs,
            )
        else:
            # Standard OpenAI client (used for platform OpenAI, new Azure format, and Azure Foundry)
            # The SDK expects base_url to be the base (e.g., https://api.openai.com/v1)
            # For new Azure format: https://{resource}.openai.azure.com/openai/v1
            # For Azure Foundry: https://{resource}.models.ai.azure.com/v1
            # If the endpoint includes API-specific paths, we need to strip them because the SDK
            # will automatically append the correct path for each API call
            base_url = self._endpoint
            if base_url.endswith("/chat/completions"):
                base_url = base_url[:-len("/chat/completions")]
            elif base_url.endswith("/responses"):
                base_url = base_url[:-len("/responses")]
            
            # For Azure Foundry endpoints (*.models.ai.azure.com), ensure they end with /v1
            from urllib.parse import urlparse
            parsed = urlparse(base_url)
            if ".models.ai.azure.com" in parsed.netloc and not base_url.endswith("/v1"):
                base_url = base_url.rstrip("/") + "/v1"
            
            # For new Azure format with Entra auth, pass token provider as api_key
            api_key_value = self._api_key
            if is_azure_new_format and self._use_entra_auth and self._azure_auth:
                # Token provider callable that the SDK will call to get bearer tokens
                # Use the Azure SDK's async get_bearer_token_provider for proper token management
                # This returns an async callable that the OpenAI SDK can await natively
                scope = get_default_scope(self._endpoint)
                api_key_value = get_async_token_provider_from_default_azure_credential(scope)
            
            self._async_client = AsyncOpenAI(
                base_url=base_url,
                api_key=api_key_value,
                **httpx_kwargs,
            )

    @abstractmethod
    def _set_openai_env_configuration_vars(self) -> None:
        """
        Sets deployment_environment_variable, endpoint_environment_variable,
        and api_key_environment_variable which are read from .env file.
        """
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
