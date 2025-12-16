# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import logging
from typing import Any, Dict, Literal, Optional

from pyrit.exceptions import (
    EmptyResponseException,
    pyrit_target_retry,
)
from pyrit.models import (
    Message,
    PromptDataType,
    construct_response_from_request,
    data_serializer_factory,
)
from pyrit.prompt_target import OpenAITarget, limit_requests_per_minute

logger = logging.getLogger(__name__)


class OpenAIImageTarget(OpenAITarget):
    """A target for image generation using OpenAI's image models."""

    def __init__(
        self,
        image_size: Literal["256x256", "512x512", "1024x1024", "1536x1024", "1024x1536"] = "1024x1024",
        quality: Optional[Literal["standard", "hd", "low", "medium", "high"]] = None,
        style: Optional[Literal["natural", "vivid"]] = None,
        *args,
        **kwargs,
    ):
        """
        Initialize the image target with specified parameters.

        Args:
            model_name (str, Optional): The name of the model.
                If no value is provided, the OPENAI_IMAGE_MODEL environment variable will be used.
            endpoint (str, Optional): The target URL for the OpenAI service.
            api_key (str | Callable[[], str], Optional): The API key for accessing the OpenAI service,
                or a callable that returns an access token. For Azure endpoints with Entra authentication,
                pass a token provider from pyrit.auth (e.g., get_azure_openai_auth(endpoint)).
                Defaults to the `OPENAI_IMAGE_API_KEY` environment variable.
            headers (str, Optional): Headers of the endpoint (JSON).
            max_requests_per_minute (int, Optional): Number of requests the target can handle per
                minute before hitting a rate limit. The number of requests sent to the target
                will be capped at the value provided.
            image_size (Literal["256x256", "512x512", "1024x1024"], Optional): The size of the generated images.
                Defaults to "1024x1024".
            quality (Literal["standard", "hd", "low", "medium", "high"], Optional): The quality of the generated images.
                Different models support different quality settings.
                For DALL-E-3, there's "standard" and "hd".
                For newer models, there are "low", "medium", and "high".
                Default is to not specify.
            style (Literal["natural", "vivid"], Optional): The style of the generated images.
                Default is to not specify.
            *args: Additional positional arguments to be passed to AzureOpenAITarget.
            **kwargs: Additional keyword arguments to be passed to AzureOpenAITarget.
            httpx_client_kwargs (dict, Optional): Additional kwargs to be passed to the
                `httpx.AsyncClient()` constructor.
                For example, to specify a 3 minutes timeout: httpx_client_kwargs={"timeout": 180}
        """
        self.quality = quality
        self.style = style
        self.image_size = image_size
        # Flag to track if we need to explicitly request b64_json format
        # Will be set to True if the model returns URLs instead of base64
        self._requires_response_format = False

        super().__init__(*args, **kwargs)

    def _set_openai_env_configuration_vars(self):
        self.model_name_environment_variable = "OPENAI_IMAGE_MODEL"
        self.endpoint_environment_variable = "OPENAI_IMAGE_ENDPOINT"
        self.api_key_environment_variable = "OPENAI_IMAGE_API_KEY"

    def _get_target_api_paths(self) -> list[str]:
        """Return API paths that should not be in the URL."""
        return ["/images/generations", "/v1/images/generations"]

    def _get_provider_examples(self) -> dict[str, str]:
        """Return provider-specific example URLs."""
        return {
            ".openai.azure.com": "https://{resource}.openai.azure.com/openai/v1",
            "api.openai.com": "https://api.openai.com/v1",
        }

    @limit_requests_per_minute
    @pyrit_target_retry
    async def send_prompt_async(
        self,
        *,
        message: Message,
    ) -> list[Message]:
        """
        Send a prompt to the DALL-E target and return the response.

        Args:
            message (Message): The message to send.

        Returns:
            list[Message]: A list containing the response from the image target.
        """
        self._validate_request(message=message)
        message_piece = message.message_pieces[0]

        logger.info(f"Sending the following prompt to the prompt target: {message_piece}")

        # Construct request parameters
        image_generation_args: Dict[str, Any] = {
            "model": self._model_name,
            "prompt": message_piece.converted_value,
            "size": self.image_size,
        }

        # Add response_format if we've detected the model returns URLs by default
        if self._requires_response_format:
            image_generation_args["response_format"] = "b64_json"

        if self.quality:
            image_generation_args["quality"] = self.quality
        if self.style:
            image_generation_args["style"] = self.style

        # Use unified error handler for consistent error handling
        response = await self._handle_openai_request(
            api_call=lambda: self._async_client.images.generate(**image_generation_args),
            request=message,
        )
        return [response]

    async def _construct_message_from_response(self, response: Any, request: Any) -> Message:
        """
        Construct a Message from an ImagesResponse.

        Args:
            response: The ImagesResponse from OpenAI SDK.
            request: The original request MessagePiece.

        Returns:
            Message: Constructed message with image path.

        Note:
            PyRIT expects base64-encoded images. Some models (like dall-e) return URLs by default,
            while others (like gpt-image-1) always return base64. This method detects the format
            and adapts automatically.
        """
        image_data = response.data[0]

        # Try to get base64 data first (preferred format)
        b64_data = getattr(image_data, "b64_json", None)

        if not b64_data:
            # Check if URL format was returned instead
            image_url = getattr(image_data, "url", None)
            if image_url:
                # Model returned URL instead of base64 - set flag and retry
                logger.info(
                    "Image model returned URL instead of base64. "
                    "Setting flag to request b64_json format in future calls."
                )
                self._requires_response_format = True
                raise EmptyResponseException(
                    message="Image was returned as URL instead of base64. Retrying with response_format parameter."
                )
            else:
                # Neither URL nor base64 - truly empty response
                raise EmptyResponseException(message="The image generation returned an empty response.")

        # Save the image and get the file path
        data = data_serializer_factory(category="prompt-memory-entries", data_type="image_path")
        await data.save_b64_image(data=b64_data)
        resp_text = data.value
        response_type: PromptDataType = "image_path"

        return construct_response_from_request(
            request=request, response_text_pieces=[resp_text], response_type=response_type
        )

    def _validate_request(self, *, message: Message) -> None:
        n_pieces = len(message.message_pieces)
        if n_pieces != 1:
            raise ValueError(f"This target only supports a single message piece. Received: {n_pieces} pieces.")

        piece_type = message.message_pieces[0].converted_value_data_type
        if piece_type != "text":
            raise ValueError(f"This target only supports text prompt input. Received: {piece_type}.")

    def is_json_response_supported(self) -> bool:
        """Check if the target supports JSON as a response format."""
        return False
