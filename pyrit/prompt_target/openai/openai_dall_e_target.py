# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import json
import logging
from typing import Any, Dict, Literal

import httpx

from pyrit.common import net_utility
from pyrit.exceptions import (
    EmptyResponseException,
    handle_bad_request_exception,
    pyrit_target_retry,
)
from pyrit.exceptions.exception_classes import RateLimitException
from pyrit.models import (
    PromptDataType,
    PromptRequestResponse,
    construct_response_from_request,
    data_serializer_factory,
)
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
            model_name (str, Optional): The name of the model.
            endpoint (str, Optional): The target URL for the OpenAI service.
            api_key (str, Optional): The API key for accessing the Azure OpenAI service.
                Defaults to the OPENAI_DALLE_API_KEY environment variable.
            headers (str, Optional): Headers of the endpoint (JSON).
            use_aad_auth (bool, Optional): When set to True, user authentication is used
                instead of API Key. DefaultAzureCredential is taken for
                https://cognitiveservices.azure.com/.default . Please run `az login` locally
                to leverage user AuthN.
            api_version (str, Optional): The version of the Azure OpenAI API. Defaults to
                "2024-06-01".
            max_requests_per_minute (int, Optional): Number of requests the target can handle per
                minute before hitting a rate limit. The number of requests sent to the target
                will be capped at the value provided.
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
            httpx_client_kwargs (dict, Optional): Additional kwargs to be passed to the
                httpx.AsyncClient() constructor.
                For example, to specify a 3 minutes timeout: httpx_client_kwargs={"timeout": 180}

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

    def _set_openai_env_configuration_vars(self):
        self.model_name_environment_variable = "OPENAI_DALLE_MODEL"
        self.endpoint_environment_variable = "OPENAI_DALLE_ENDPOINT"
        self.api_key_environment_variable = "OPENAI_DALLE_API_KEY"

    @limit_requests_per_minute
    @pyrit_target_retry
    async def send_prompt_async(
        self,
        *,
        prompt_request: PromptRequestResponse,
    ) -> PromptRequestResponse:
        """
        Send a prompt to the DALL-E target and return the response.

        Args:
            prompt_request (PromptRequestResponse): The prompt request to send.

        Returns:
            PromptRequestResponse: The response from the DALL-E target.
        """
        self._validate_request(prompt_request=prompt_request)
        request = prompt_request.request_pieces[0]

        logger.info(f"Sending the following prompt to the prompt target: {request}")

        # Refresh auth headers if using AAD
        self.refresh_auth_headers()

        body = self._construct_request_body(prompt=request.converted_value)

        params = {}
        if self._api_version is not None:
            params["api-version"] = self._api_version

        try:
            http_response: httpx.Response = await net_utility.make_request_and_raise_if_error_async(
                endpoint_uri=self._endpoint,
                method="POST",
                headers=self._headers,
                request_body=body,
                params=params,
                **self._httpx_client_kwargs,
            )
        except httpx.HTTPStatusError as StatusError:
            if StatusError.response.status_code == 400:
                # Handle Bad Request
                return handle_bad_request_exception(response_text=StatusError.response.text, request=request)
            elif StatusError.response.status_code == 429:
                raise RateLimitException()
            else:
                raise

        json_response = json.loads(http_response.text)
        b64_data = json_response["data"][0]["b64_json"]

        # Handle empty response using retry
        if not b64_data:
            raise EmptyResponseException(message="The chat returned an empty response.")

        data = data_serializer_factory(category="prompt-memory-entries", data_type="image_path")
        await data.save_b64_image(data=b64_data)
        resp_text = data.value
        response_type: PromptDataType = "image_path"

        response_entry = construct_response_from_request(
            request=request, response_text_pieces=[resp_text], response_type=response_type
        )

        return response_entry

    def _construct_request_body(self, prompt: str):
        image_generation_args: Dict[str, Any] = {
            "model": self._model_name,
            "prompt": prompt,
            "n": self.num_images,
            "size": self.image_size,
            "response_format": "b64_json",
        }
        if self.dalle_version == "dall-e-3" and self.quality and self.style:
            image_generation_args["quality"] = self.quality
            image_generation_args["style"] = self.style

        return image_generation_args

    def _validate_request(self, *, prompt_request: PromptRequestResponse) -> None:
        if len(prompt_request.request_pieces) != 1:
            raise ValueError("This target only supports a single prompt request piece.")

        if prompt_request.request_pieces[0].converted_value_data_type != "text":
            raise ValueError("This target only supports text prompt input.")

    def is_json_response_supported(self) -> bool:
        """Indicates that this target supports JSON response format."""
        return False
