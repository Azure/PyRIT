# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import json
import logging
import pathlib
import concurrent.futures
import asyncio
from typing import Literal, Optional

from openai import BadRequestError

from pyrit.common.path import RESULTS_PATH
from pyrit.memory.memory_interface import MemoryInterface
from pyrit.models import PromptRequestResponse
from pyrit.models.prompt_request_piece import PromptRequestPiece, PromptResponseError
from pyrit.prompt_target import PromptTarget
from pyrit.prompt_target.prompt_chat_target.openai_chat_target import AzureOpenAIChatTarget
from pyrit.prompt_normalizer import data_serializer_factory

logger = logging.getLogger(__name__)


class DALLETarget(PromptTarget):
    """
    The Dalle3Target takes a prompt and generates images
    This class initializes a DALL-E image target

    Args:
        deployment_name (str): The name of the deployment.
        endpoint (str): The endpoint URL for the service.
        api_key (str): The API key for accessing the service.
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
        self.image_target = AzureOpenAIChatTarget(
            deployment_name=deployment_name, endpoint=endpoint, api_key=api_key, api_version=api_version
        )
        self.output_dir = pathlib.Path(RESULTS_PATH) / "images"

        self.headers = headers

    def send_prompt(
        self,
        *,
        prompt_request: PromptRequestResponse,
    ) -> PromptRequestResponse:
        """
        Deprecated. Use send_prompt_async instead.
        """
        pool = concurrent.futures.ThreadPoolExecutor()
        return pool.submit(asyncio.run, self.send_prompt_async(prompt_request=prompt_request)).result()

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

        self.validate_request(prompt_request=prompt_request)
        request = prompt_request.request_pieces[0]

        self._memory.add_request_response_to_memory(request=prompt_request)

<<<<<<< HEAD
        return await self._generate_images_async(prompt=request.converted_value, request=request)
=======
        return await self._generate_images_async(prompt=request.converted_prompt_text, request=request)
>>>>>>> main

    async def _generate_images_async(self, prompt: str, request=PromptRequestPiece) -> PromptRequestResponse:
        try:
            if self.dalle_version == "dall-e-3":
                if self.quality and self.style:
                    response = await self.image_target._async_client.images.generate(
                        model=self.deployment_name,
                        prompt=prompt,
                        n=self.n,
                        size=self.image_size,
                        response_format="b64_json",
                        quality=self.quality,
                        style=self.style,
                    )
                else:
                    response = await self.image_target._async_client.images.generate(
                        model=self.deployment_name,
                        prompt=prompt,
                        n=self.n,
                        size=self.image_size,
                        response_format="b64_json",
                    )
            else:
                response = await self.image_target._async_client.images.generate(
                    model=self.deployment_name,
                    prompt=prompt,
                    n=self.n,
                    size=self.image_size,
                    response_format="b64_json",
                )
            json_response = json.loads(response.model_dump_json())

            data = data_serializer_factory(data_type="image_path")
            b64_data = json_response["data"][0]["b64_json"]
            data.save_b64_image(data=b64_data)
            prompt_text = data.prompt_text
            error: PromptResponseError = "none"

        except BadRequestError as e:
            json_response = {"exception type": "Blocked", "data": ""}
            json_response["error"] = e.body
            prompt_text = "content blocked"
            error = "blocked"

        except json.JSONDecodeError as e:
            json_response = {"error": e, "exception type": "JSON Error"}
            prompt_text = "JSON Error"
            error = "processing"

        except Exception as e:
            json_response = {"error": e, "exception type": "exception"}
            prompt_text = "target error"
            error = "unknown"

        return self._memory.add_response_entries_to_memory(
            request=request,
            response_text_pieces=[prompt_text],
            response_type="image_path",
            prompt_metadata=json.dumps(json_response),
            error=error,
        )

    def validate_request(self, *, prompt_request: PromptRequestResponse) -> None:
        if len(prompt_request.request_pieces) != 1:
            raise ValueError("This target only supports a single prompt request piece.")

<<<<<<< HEAD
        if prompt_request.request_pieces[0].converted_value_data_type != "text":
=======
        if prompt_request.request_pieces[0].converted_prompt_data_type != "text":
>>>>>>> main
            raise ValueError("This target only supports text prompt input.")

        request = prompt_request.request_pieces[0]
        messages = self._memory.get_chat_messages_with_conversation_id(conversation_id=request.conversation_id)

        if len(messages) > 0:
            raise ValueError("This target only supports a single turn conversation.")
