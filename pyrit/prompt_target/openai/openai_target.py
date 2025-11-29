# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import json
import logging
import os
import re
from abc import abstractmethod
from typing import Any, Callable, Optional, Union
from urllib.parse import parse_qs, urlparse

from openai import (
    AsyncAzureOpenAI,
    AsyncOpenAI,
    BadRequestError,
    ContentFilterFinishReasonError,
    RateLimitError,
)
from openai._exceptions import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AuthenticationError,
)

from pyrit.auth import AzureAuth
from pyrit.auth.azure_auth import (
    get_async_token_provider_from_default_azure_credential,
    get_default_scope,
)
from pyrit.common import default_values
from pyrit.exceptions.exception_classes import (
    RateLimitException,
    handle_bad_request_exception,
)
from pyrit.models import Message, MessagePiece
from pyrit.prompt_target import PromptChatTarget
from pyrit.prompt_target.openai.openai_error_handling import (
    _extract_error_payload,
    _extract_request_id_from_exception,
    _extract_retry_after_from_exception,
)

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
            self._api_key = None
        else:
            self._api_key = default_values.get_non_required_value(
                env_var_name=self.api_key_environment_variable, passed_value=passed_api_key
            )

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
        match = re.search(r"/deployments/([^/]+)/", url)
        if match:
            deployment = match.group(1)
            logger.info(f"Extracted deployment name from URL: {deployment}")
            return deployment

        return ""

    def refresh_auth_headers(self) -> None:
        """
        Refresh the authentication headers. This is particularly useful for Entra authentication
        where tokens need to be refreshed periodically.
        
        Note: For SDK-based targets, token refresh is handled automatically by the SDK's
        token provider mechanism. This method is kept for backward compatibility.
        """
        if self._azure_auth:
            # Token refresh is now handled automatically by the SDK's token provider
            # This explicit refresh is kept for any edge cases or manual refresh needs
            self._azure_auth.refresh_token()

    def _initialize_azure_openai_old_format(self, *, httpx_kwargs: dict) -> None:
        """
        Initialize AsyncAzureOpenAI client for old Azure format.

        Old Azure format: https://{resource}.openai.azure.com/openai/deployments/{deployment}/...?api-version=...
        Uses AsyncAzureOpenAI client with api-version parameter (supports both API key and Entra auth)

        Note: This method can be removed once old Azure format is no longer supported.
        """
        # Extract API version from query parameter if present
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
        # Merge custom headers with httpx_client_kwargs
        httpx_kwargs = self._httpx_client_kwargs.copy()
        if self._headers:
            httpx_kwargs.setdefault("default_headers", {}).update(self._headers)

        # Determine if this is Azure OpenAI based on the endpoint
        is_azure = "azure" in self._endpoint.lower() if self._endpoint else False

        # Check if it's the new Azure format (OpenAI-compatible)
        # New format includes:
        # - https://{resource}.openai.azure.com/openai/v1
        # - https://{resource}.models.ai.azure.com/... (Azure Foundry endpoints)
        is_azure_new_format = False

        # Only old Azure format uses AsyncAzureOpenAI
        # Everything else (platform OpenAI, new Azure format, Azure Foundry) uses standard AsyncOpenAI
        if is_azure:
            parsed_url = urlparse(self._endpoint)
            # New format has /openai/v1 path OR uses models.ai.azure.com domain
            is_azure_new_format = "/openai/v1" in parsed_url.path or ".models.ai.azure.com" in parsed_url.netloc

        if is_azure and not is_azure_new_format:
            # Old Azure format - delegate to separate method for easy removal later
            self._initialize_azure_openai_old_format(httpx_kwargs=httpx_kwargs)
        else:
            # Standard OpenAI client (used for platform OpenAI, new Azure format, and Azure Foundry)
            # The SDK expects base_url to be the base (e.g., https://api.openai.com/v1)
            # For new Azure format: https://{resource}.openai.azure.com/openai/v1
            # For Azure Foundry: https://{resource}.models.ai.azure.com/v1
            # If the endpoint includes API-specific paths, we need to strip them because the SDK
            # will automatically append the correct path for each API call
            base_url = self._endpoint
            if base_url.endswith("/chat/completions"):
                base_url = base_url[: -len("/chat/completions")]
            elif base_url.endswith("/completions"):
                base_url = base_url[: -len("/completions")]
            elif base_url.endswith("/responses"):
                base_url = base_url[: -len("/responses")]
            elif base_url.endswith("/images/generations"):
                base_url = base_url[: -len("/images/generations")]
            elif base_url.endswith("/audio/speech"):
                base_url = base_url[: -len("/audio/speech")]
            elif base_url.endswith("/v1/videos") or base_url.endswith("/videos"):
                # Strip videos path for video API
                if base_url.endswith("/v1/videos"):
                    base_url = base_url[: -len("/videos")]  # Keep /v1
                else:
                    base_url = base_url[: -len("/videos")]

            # For Azure Foundry endpoints (*.models.ai.azure.com), ensure they end with /v1
            parsed = urlparse(base_url)
            if ".models.ai.azure.com" in parsed.netloc and not base_url.endswith("/v1"):
                base_url = base_url.rstrip("/") + "/v1"

            # For new Azure format with Entra auth, pass token provider as api_key
            api_key_value: Any = self._api_key
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

    async def _handle_openai_request(
        self,
        *,
        api_call: Callable,
        request: Message,
    ) -> Message:
        """
        Unified error handling wrapper for all OpenAI SDK requests.

        This method wraps any OpenAI SDK call and handles all common error scenarios:
        - Content filtering (both proactive checks and SDK exceptions)
        - Bad request errors (400s with content filter detection)
        - Rate limiting (429s with retry-after extraction)
        - API status errors (other HTTP errors)
        - Transient errors (timeouts, connection issues)
        - Authentication errors

        Automatically detects the response type and applies appropriate validation and content
        filter checks via abstract methods. On success, constructs and returns a Message object.

        Args:
            api_call: Async callable that invokes the OpenAI SDK method.
            request: The Message representing the user's request (for error responses).

        Returns:
            Message: The constructed message response (success or error).

        Raises:
            RateLimitException: For 429 rate limit errors.
            Various OpenAI SDK exceptions: For non-recoverable errors.
        """
        try:
            # Execute the API call
            response = await api_call()

            # Extract MessagePiece for validation and construction (most targets use single piece)
            request_piece = request.message_pieces[0] if request.message_pieces else None

            # Check for content filter via subclass implementation
            if self._check_content_filter(response):
                return self._handle_content_filter_response(response, request_piece)

            # Validate response via subclass implementation
            error_message = self._validate_response(response, request_piece)
            if error_message:
                return error_message

            # Construct and return Message from validated response
            return await self._construct_message_from_response(response, request_piece)

        except ContentFilterFinishReasonError as e:
            # Content filter error raised by SDK during parse/structured output flows
            request_id = _extract_request_id_from_exception(e)
            logger.error(f"Content filter error (SDK raised). request_id={request_id} error={e}")

            # Convert exception to response-like object for consistent handling
            error_str = str(e)

            class _ErrorResponse:
                def model_dump_json(self):
                    return error_str

            request_piece = request.message_pieces[0] if request.message_pieces else None
            return self._handle_content_filter_response(_ErrorResponse(), request_piece)
        except BadRequestError as e:
            # Handle 400 errors - includes input policy filters and some Azure output-filter 400s
            payload, is_content_filter = _extract_error_payload(e)
            request_id = _extract_request_id_from_exception(e)

            # Safely serialize payload for logging
            try:
                payload_str = payload if isinstance(payload, str) else json.dumps(payload)[:200]
            except (TypeError, ValueError):
                # If JSON serialization fails (e.g., contains non-serializable objects), use str()
                payload_str = str(payload)[:200]

            logger.warning(
                f"BadRequestError request_id={request_id} is_content_filter={is_content_filter} "
                f"payload={payload_str}"
            )

            request_piece = request.message_pieces[0] if request.message_pieces else None
            return handle_bad_request_exception(
                response_text=str(payload),
                request=request_piece,
                error_code=400,
                is_content_filter=is_content_filter,
            )
        except RateLimitError as e:
            # SDK's RateLimitError (429)
            request_id = _extract_request_id_from_exception(e)
            retry_after = _extract_retry_after_from_exception(e)
            logger.warning(f"RateLimitError request_id={request_id} retry_after={retry_after} error={e}")
            raise RateLimitException()
        except APIStatusError as e:
            # Other API status errors - check for 429 here as well
            request_id = _extract_request_id_from_exception(e)
            if getattr(e, "status_code", None) == 429:
                retry_after = _extract_retry_after_from_exception(e)
                logger.warning(f"429 via APIStatusError request_id={request_id} retry_after={retry_after}")
                raise RateLimitException()
            else:
                logger.exception(
                    f"APIStatusError request_id={request_id} status={getattr(e, 'status_code', None)} error={e}"
                )
                raise
        except (APITimeoutError, APIConnectionError) as e:
            # Transient infrastructure errors - these are retryable
            request_id = _extract_request_id_from_exception(e)
            logger.warning(f"Transient API error ({e.__class__.__name__}) request_id={request_id} error={e}")
            raise
        except AuthenticationError as e:
            # Authentication errors - non-retryable, surface quickly
            request_id = _extract_request_id_from_exception(e)
            logger.error(f"Authentication error request_id={request_id} error={e}")
            raise

    @abstractmethod
    async def _construct_message_from_response(self, response: Any, request: MessagePiece) -> Message:
        """
        Construct a Message from the OpenAI SDK response.

        This method extracts the relevant data from the SDK response object and
        constructs a Message with appropriate message pieces. It may include
        async operations like saving files for image/audio/video responses.

        Args:
            response: The response object from OpenAI SDK (e.g., ChatCompletion, Response, etc.).
            request: The original request MessagePiece.

        Returns:
            Message: Constructed message with extracted content.
        """
        pass

    def _check_content_filter(self, response: Any) -> bool:
        """
        Check if the response indicates content filtering.

        Override this method in subclasses that need content filter detection.
        Default implementation returns False (no content filter).

        Args:
            response: The response object from OpenAI SDK.

        Returns:
            bool: True if content filter detected, False otherwise.
        """
        return False

    def _handle_content_filter_response(self, response: Any, request: MessagePiece) -> Message:
        """
        Handle content filter errors by creating a proper error Message.

        Args:
            response: The response object from OpenAI SDK.
            request: The original request message piece.

        Returns:
            Message object with error type indicating content was filtered.
        """
        logger.warning("Output content filtered by content policy.")
        return handle_bad_request_exception(
            response_text=response.model_dump_json(),
            request=request,
            error_code=200,
            is_content_filter=True,
        )

    def _validate_response(self, response: Any, request: MessagePiece) -> Optional[Message]:
        """
        Validate the response and return error Message if needed.

        Override this method in subclasses that need custom response validation.
        Default implementation returns None (no validation errors).

        Args:
            response: The response object from OpenAI SDK.
            request: The original request MessagePiece.

        Returns:
            Optional[Message]: Error Message if validation fails, None otherwise.

        Raises:
            Various exceptions for validation failures.
        """
        return None

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
