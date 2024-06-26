# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import json
import logging
import pathlib

from typing import Literal, Optional, Dict, Any
from openai import BadRequestError

from pyrit.common.path import RESULTS_PATH
from pyrit.exceptions import EmptyResponseException, pyrit_target_retry, handle_bad_request_exception
from pyrit.memory.memory_interface import MemoryInterface
from pyrit.models import PromptRequestResponse, data_serializer_factory, construct_response_from_request, PromptDataType
from pyrit.prompt_target import AzureOpenAIChatTarget, PromptTarget

logger = logging.getLogger(__name__)


class DALLETarget(PromptTarget):
    """
    The Dalle3Target takes a prompt and generates images
    This class initializes a DALL-E image target

    Args:
        deployment_name (str): The name of the deployment.
        endpoint (str): The endpoint URL for the service.
        api_key (str): The API key for accessing the service.
        use_aad_auth (bool, optional): When set to True, user authentication is used
            instead of API Key. DefaultAzureCredential is taken for
            https://cognitiveservices.azure.com/.default. Please run `az login` locally
            to leverage user AuthN.
        api_version (str, optional): The API version. Defaults to "2024-02-01".
        image_size (str, optional): The size of the image to output, must be a value of VALID_SIZES.
            Defaults to 1024x1024.
        num_images (int, optional): The number of output images to generate.
            Defaults to 1. For DALLE-3, can only be 1, for DALLE-2 max is 10 images.
        dalle_version (int, optional): Version of DALLE service. Defaults to 3.
        memory: (memory, optional): Memory to store the chat messages. DuckDBMemory will be used by default.
        headers (dict, optional): Headers of the endpoint.
        quality (str, optional): picture quality. Defaults to standard
        style (str, optional): image style. Defaults to natural
    """

    def __init__(
        self,
        *,
        deployment_name: str = None,
        endpoint: str = None,
        api_key: str = None,
        use_aad_auth: bool = False,
        api_version: str = "2024-02-01",
        image_size: Literal["256x256", "512x512", "1024x1024"] = "1024x1024",
        num_images: int = 1,
        dalle_version: Literal["dall-e-2", "dall-e-3"] = "dall-e-2",
        memory: MemoryInterface | None = None,
        headers: Optional[dict[str, str]] = None,
        quality: Literal["standard", "hd"] = "standard",
        style: Literal["natural", "vivid"] = "natural",
    ):

        super().__init__(memory=memory)

        # make sure number of images and headers are allowed by Dall-e version
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
        self.n = num_images

        self.deployment_name = deployment_name
        self.output_dir = pathlib.Path(RESULTS_PATH) / "images"
        self.headers = headers

        target_kwargs: Dict[str, Any] = {
            "deployment_name": deployment_name,
            "endpoint": endpoint,
            "api_version": api_version,
        }
        if use_aad_auth:
            target_kwargs["use_aad_auth"] = True
        else:
            target_kwargs["api_key"] = api_key
        self._image_target = AzureOpenAIChatTarget(**target_kwargs)

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
            "model": self.deployment_name,
            "prompt": prompt,
            "n": self.n,
            "size": self.image_size,
            "response_format": "b64_json",
        }
        if self.dalle_version == "dall-e-3" and self.quality and self.style:
            image_generation_args["quality"] = self.quality
            image_generation_args["style"] = self.style

        try:
            b64_data = await self._generate_image_response_async(image_generation_args)
            data = data_serializer_factory(data_type="image_path")
            data.save_b64_image(data=b64_data)
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
        result = await self._image_target._async_client.images.generate(**image_generation_args)
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
