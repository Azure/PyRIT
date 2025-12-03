# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import logging
from typing import Any, Dict, Literal
from urllib.parse import urlparse

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
        image_size: Literal["256x256", "512x512", "1024x1024"] = "1024x1024",
        num_images: int = 1,
        image_version: Literal["dall-e-2", "dall-e-3"] = "dall-e-2",
        quality: Literal["standard", "hd"] = "standard",
        style: Literal["natural", "vivid"] = "natural",
        *args,
        **kwargs,
    ):
        """
        Initialize the image target with specified parameters.

        Args:
            model_name (str, Optional): The name of the model.
            endpoint (str, Optional): The target URL for the OpenAI service.
            api_key (str, Optional): The API key for accessing the Azure OpenAI service.
                Defaults to the `OPENAI_IMAGE_API_KEY` environment variable.
            headers (str, Optional): Headers of the endpoint (JSON).
            use_entra_auth (bool, Optional): When set to True, user authentication is used
                instead of API Key. DefaultAzureCredential is taken for
                https://cognitiveservices.azure.com/.default . Please run `az login` locally
                to leverage user AuthN.
            max_requests_per_minute (int, Optional): Number of requests the target can handle per
                minute before hitting a rate limit. The number of requests sent to the target
                will be capped at the value provided.
            image_size (Literal["256x256", "512x512", "1024x1024"], Optional): The size of the generated images.
                Defaults to "1024x1024".
            num_images (int, Optional): The number of images to generate. Defaults to 1. For DALL-E-2, this can be
                between 1 and 10. For DALL-E-3, this must be 1.
            image_version (Literal["dall-e-2", "dall-e-3"], Optional): The version of DALL-E to use. Defaults to
                "dall-e-2".
            quality (Literal["standard", "hd"], Optional): The quality of the generated images. Only applicable for
                DALL-E-3. Defaults to "standard".
            style (Literal["natural", "vivid"], Optional): The style of the generated images. Only applicable for
                DALL-E-3. Defaults to "natural".
            *args: Additional positional arguments to be passed to AzureOpenAITarget.
            **kwargs: Additional keyword arguments to be passed to AzureOpenAITarget.
            httpx_client_kwargs (dict, Optional): Additional kwargs to be passed to the
                `httpx.AsyncClient()` constructor.
                For example, to specify a 3 minutes timeout: httpx_client_kwargs={"timeout": 180}

        Raises:
            ValueError: If `num_images` is not 1 for DALL-E-3.
            ValueError: If `num_images` is less than 1 or greater than 10 for DALL-E-2.
        """
        self.image_version = image_version
        if image_version == "dall-e-3":
            if num_images != 1:
                raise ValueError("DALL-E-3 can only generate 1 image at a time.")
            self.quality = quality
            self.style = style
        elif image_version == "dall-e-2":
            if num_images < 1 or num_images > 10:
                raise ValueError("DALL-E-2 can generate only up to 10 images at a time.")

        self.image_size = image_size
        self.num_images = num_images

        super().__init__(*args, **kwargs)

    def _set_openai_env_configuration_vars(self):
        self.model_name_environment_variable = "OPENAI_IMAGE_MODEL"
        self.endpoint_environment_variable = "OPENAI_IMAGE_ENDPOINT"
        self.api_key_environment_variable = "OPENAI_IMAGE_API_KEY"

    def _normalize_url_for_target(self, base_url: str) -> str:
        """
        Normalize and validate the URL for image generation.

        Strips /images/generations if present (for all endpoints, since the SDK constructs the path).

        Args:
            base_url: The endpoint URL to normalize.

        Returns:
            The normalized URL.
        """
        # Validate URL format first, before any modifications
        image_url_patterns = [
            r"/v1$",
            r"/images/generations",
            r"/deployments/[^/]+/",
            r"openai/v1",
            r"\.models\.ai\.azure\.com",
        ]
        self._warn_if_irregular_endpoint(image_url_patterns)

        # Strip images/generations path if present (SDK will add it back)
        if base_url.endswith("/images/generations"):
            base_url = base_url[: -len("/images/generations")]

        return base_url

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
            "n": self.num_images,
            "size": self.image_size,
            "response_format": "b64_json",
        }
        if self.image_version == "dall-e-3" and self.quality and self.style:
            image_generation_args["quality"] = self.quality
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
        """
        # Extract base64 image data from response
        b64_data = response.data[0].b64_json

        # Handle empty response using retry
        if not b64_data:
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
        """Indicates that this target supports JSON response format."""
        return False
