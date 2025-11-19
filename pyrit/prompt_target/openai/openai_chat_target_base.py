# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import json
import logging
from typing import Any, MutableSequence, Optional

from openai import (
    BadRequestError,
    RateLimitError,
    ContentFilterFinishReasonError,
    APIStatusError,
)

from pyrit.exceptions import (
    PyritException,
    handle_bad_request_exception,
    pyrit_target_retry,
)
from pyrit.exceptions.exception_classes import RateLimitException
from pyrit.models import (
    Message,
    MessagePiece,
)
from pyrit.prompt_target import (
    OpenAITarget,
    PromptChatTarget,
    limit_requests_per_minute,
)

logger = logging.getLogger(__name__)


class OpenAIChatTargetBase(OpenAITarget, PromptChatTarget):
    """
    This is the base class for multimodal (image and text) input and text output generation.

    It provides the foundation for the chat completions and response API classes.
    """

    def __init__(
        self,
        *,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        is_json_supported: bool = True,
        extra_body_parameters: Optional[dict[str, Any]] = None,
        **kwargs,
    ):
        """
        Initialize the OpenAIChatTargetBase with the given parameters.

        Args:
            model_name (str, Optional): The name of the model.
            endpoint (str, Optional): The target URL for the OpenAI service.
            api_key (str, Optional): The API key for accessing the Azure OpenAI service.
                Defaults to the OPENAI_CHAT_KEY environment variable.
            headers (str, Optional): Headers of the endpoint (JSON).
            use_entra_auth (bool, Optional): When set to True, user authentication is used
                instead of API Key. DefaultAzureCredential is taken for
                https://cognitiveservices.azure.com/.default . Please run `az login` locally
                to leverage user AuthN.
            max_requests_per_minute (int, Optional): Number of requests the target can handle per
                minute before hitting a rate limit. The number of requests sent to the target
                will be capped at the value provided.
            temperature (float, Optional): The temperature parameter for controlling the
                randomness of the response.
            top_p (float, Optional): The top-p parameter for controlling the diversity of the
                response.
            is_json_supported (bool, Optional): If True, the target will support formatting responses as JSON by
                setting the response_format header. Official OpenAI models all support this, but if you are using
                this target with different models, is_json_supported should be set correctly to avoid issues when
                using adversarial infrastructure (e.g. Crescendo scorers will set this flag).
            extra_body_parameters (dict, Optional): Additional parameters to be included in the request body.
            httpx_client_kwargs (dict, Optional): Additional kwargs to be passed to the
                httpx.AsyncClient() constructor.
                For example, to specify a 3 minute timeout: httpx_client_kwargs={"timeout": 180}

        Raises:
            PyritException: If the temperature or top_p values are out of bounds.
            ValueError: If the temperature is not between 0 and 2 (inclusive).
            ValueError: If the top_p is not between 0 and 1 (inclusive).
            RateLimitException: If the target is rate-limited.
            httpx.HTTPStatusError: If the request fails with a 400 Bad Request or 429 Too Many Requests error.
            json.JSONDecodeError: If the response from the target is not valid JSON.
            Exception: If the request fails for any other reason.
        """
        super().__init__(**kwargs)

        if temperature is not None and (temperature < 0 or temperature > 2):
            raise PyritException("temperature must be between 0 and 2 (inclusive).")
        if top_p is not None and (top_p < 0 or top_p > 1):
            raise PyritException("top_p must be between 0 and 1 (inclusive).")

        self._temperature = temperature
        self._top_p = top_p
        self._is_json_supported = is_json_supported
        self._extra_body_parameters = extra_body_parameters

    @limit_requests_per_minute
    @pyrit_target_retry
    async def send_prompt_async(self, *, message: Message) -> Message:
        """
        Asynchronously sends a message and handles the response within a managed conversation context.

        Args:
            message (Message): The message object.

        Returns:
            Message: The updated conversation entry with the response from the prompt target.
        """
        self._validate_request(message=message)

        message_piece: MessagePiece = message.message_pieces[0]

        is_json_response = self.is_response_format_json(message_piece)

        conversation = self._memory.get_conversation(conversation_id=message_piece.conversation_id)
        conversation.append(message)

        logger.info(f"Sending the following prompt to the prompt target: {message}")

        body = await self._construct_request_body(conversation=conversation, is_json_response=is_json_response)

        try:
            # Use the OpenAI SDK for making the request
            response = await self._make_chat_completion_request(body)
            
            # Convert the SDK response to our Message format
            return self._construct_message_from_completion_response(
                completion_response=response, message_piece=message_piece
            )
            
        except ContentFilterFinishReasonError as e:
            # Content filter error - this is raised by the SDK when finish_reason is "content_filter"
            logger.error(f"Content filter error: {e}")
            return handle_bad_request_exception(
                response_text=str(e),
                request=message_piece,
                error_code=200,  # Content filter with 200 status
                is_content_filter=True,
            )
        except BadRequestError as e:
            # Handle Bad Request from the SDK
            error_response_text = e.body if hasattr(e, 'body') else str(e)
            
            # Check if it's a content filter issue
            is_content_filter = False
            if isinstance(error_response_text, dict):
                is_content_filter = error_response_text.get("error", {}).get("code") == "content_filter"
            elif isinstance(error_response_text, str):
                try:
                    json_error = json.loads(error_response_text)
                    is_content_filter = json_error.get("error", {}).get("code") == "content_filter"
                except json.JSONDecodeError:
                    is_content_filter = "content_filter" in error_response_text

            return handle_bad_request_exception(
                response_text=str(error_response_text),
                request=message_piece,
                error_code=400,
                is_content_filter=is_content_filter,
            )
        except RateLimitError as e:
            # SDK's RateLimitError - convert to our exception
            logger.warning(f"Rate limit hit: {e}")
            raise RateLimitException()
        except APIStatusError as e:
            # Other API errors
            if e.status_code == 429:
                raise RateLimitException()
            else:
                raise

    async def _make_chat_completion_request(self, body: dict):
        """
        Make the actual chat completion request using the OpenAI SDK.
        This method should be overridden by subclasses to use the appropriate SDK method.
        
        Args:
            body (dict): The request body parameters.
            
        Returns:
            The completion response from the OpenAI SDK.
        """
        raise NotImplementedError

    async def _construct_request_body(self, conversation: MutableSequence[Message], is_json_response: bool) -> dict:
        """
        Construct the request body from a conversation.
        This method should be overridden by subclasses.
        
        Args:
            conversation (MutableSequence[Message]): The conversation history.
            is_json_response (bool): Whether to request JSON response format.
            
        Returns:
            dict: The request body parameters.
        """
        raise NotImplementedError

    def _construct_message_from_completion_response(
        self,
        *,
        completion_response,
        message_piece: MessagePiece,
    ) -> Message:
        """
        Construct a Message from the OpenAI SDK completion response.
        This method should be overridden by subclasses.
        
        Args:
            completion_response: The completion response from the OpenAI SDK.
            message_piece (MessagePiece): The original request message piece.
            
        Returns:
            Message: The constructed message.
        """
        raise NotImplementedError

    def is_json_response_supported(self) -> bool:
        """Indicates that this target supports JSON response format."""
        return self._is_json_supported
