# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import json
import logging

from typing import Literal, Dict, Any
from openai import BadRequestError

from pyrit.exceptions import EmptyResponseException, pyrit_target_retry, handle_bad_request_exception
from pyrit.models import PromptRequestResponse, data_serializer_factory, construct_response_from_request, PromptDataType
from pyrit.prompt_target import OpenAITarget, limit_requests_per_minute

logger = logging.getLogger(__name__)


class OpenAIDALLETarget(OpenAITarget):
    """
    The Dalle3Target takes a prompt and generates images
    This class initializes a DALL-E image target
    """

    def __init__(
        self,
        image_size: Literal["256x256", "512x512", "1024x1024"] = "1024x1024",
        num_images: int = 1,
        dalle_version: Literal["dall-e-2", "dall-e-3"] = "dall-e-2",
        quality: Literal["standard", "hd"] = "standard",
        style: Literal["natural", "vivid"] = "natural",
        *args,
        **kwargs,
    ):
        """
        Initialize the DALL-E target with specified parameters.

        Args:
            image_size (Literal["256x256", "512x512", "1024x1024"], Optional): The size of the generated images.
                Defaults to "1024x1024".
            num_images (int, Optional): The number of images to generate. Defaults to 1. For DALL-E-2, this can be
                between 1 and 10. For DALL-E-3, this must be 1.
            dalle_version (Literal["dall-e-2", "dall-e-3"], Optional): The version of DALL-E to use. Defaults to
                "dall-e-2".
            quality (Literal["standard", "hd"], Optional): The quality of the generated images. Only applicable for
                DALL-E-3. Defaults to "standard".
            style (Literal["natural", "vivid"], Optional): The style of the generated images. Only applicable for
                DALL-E-3. Defaults to "natural".
            *args: Additional positional arguments to be passed to AzureOpenAITarget.
            **kwargs: Additional keyword arguments to be passed to AzureOpenAITarget.

        Raises:
            ValueError: If `num_images` is not 1 for DALL-E-3.
            ValueError: If `num_images` is less than 1 or greater than 10 for DALL-E-2.
        """

        self.dalle_version = dalle_version
        if dalle_version == "dall-e-3":
            if num_images != 1:
                raise ValueError("DALL-E-3 can only generate 1 image at a time.")
            self.quality = quality
            self.style = style
        elif dalle_version == "dall-e-2":
            if num_images < 1 or num_images > 10:
                raise ValueError("DALL-E-2 can generate only up to 10 images at a time.")

        self.image_size = image_size
        self.num_images = num_images

        super().__init__(*args, **kwargs)

    def _set_azure_openai_env_configuration_vars(self):
        self.deployment_environment_variable = "AZURE_OPENAI_DALLE_DEPLOYMENT"
        self.endpoint_uri_environment_variable = "AZURE_OPENAI_DALLE_ENDPOINT"
        self.api_key_environment_variable = "AZURE_OPENAI_DALLE_API_KEY"

    @limit_requests_per_minute
    async def send_prompt_async(
        self,
        *,
        prompt_request: PromptRequestResponse,
    ) -> PromptRequestResponse:
        """
        (Async) Sends prompt to image target and returns response

        Args:
            prompt_request (PromptRequestResponse): the prompt to send formatted as an object

        Returns: response from target model formatted as an object
        """

        self._validate_request(prompt_request=prompt_request)
        request = prompt_request.request_pieces[0]
        prompt = request.converted_value

        image_generation_args: Dict[str, Any] = {
            "model": self._deployment_name,
            "prompt": prompt,
            "n": self.num_images,
            "size": self.image_size,
            "response_format": "b64_json",
        }
        if self.dalle_version == "dall-e-3" and self.quality and self.style:
            image_generation_args["quality"] = self.quality
            image_generation_args["style"] = self.style

        try:
            b64_data = await self._generate_image_response_async(image_generation_args)
            data = data_serializer_factory(data_type="image_path")
            await data.save_b64_image(data=b64_data)
            resp_text = data.value
            response_type: PromptDataType = "image_path"

            response_entry = construct_response_from_request(
                request=request, response_text_pieces=[resp_text], response_type=response_type
            )

        except BadRequestError as bre:
            response_entry = handle_bad_request_exception(response_text=bre.message, request=request)

        return response_entry

    @pyrit_target_retry
    async def _generate_image_response_async(self, image_generation_args):
        """
        Asynchronously generates an image using the provided generation arguments.

        Retries the function if it raises RateLimitError (HTTP 429) or EmptyResponseException,
        with a wait time between retries that follows an exponential backoff strategy.
        Logs retry attempts at the INFO level and stops after a maximum number of attempts.

        Args:
            image_generation_args (dict): The arguments required for image generation.

        Returns:
            The generated image  in base64 format.

        Raises:
            RateLimitError: If the rate limit is exceeded and the maximum number of retries is exhausted.
            EmptyResponseException: If the response is empty after exhausting the maximum number of retries.
        """
        result = await self._async_client.images.generate(**image_generation_args)
        json_response = json.loads(result.model_dump_json())
        b64_data = json_response["data"][0]["b64_json"]
        # Handle empty response using retry
        if not b64_data:
            raise EmptyResponseException(message="The chat returned an empty response.")
        return b64_data

    def _validate_request(self, *, prompt_request: PromptRequestResponse) -> None:
        if len(prompt_request.request_pieces) != 1:
            raise ValueError("This target only supports a single prompt request piece.")

        if prompt_request.request_pieces[0].converted_value_data_type != "text":
            raise ValueError("This target only supports text prompt input.")
