# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import json
import logging
import pathlib
from enum import Enum
import concurrent.futures
import asyncio

from openai import BadRequestError

from pyrit.common.path import RESULTS_PATH
from pyrit.memory.memory_interface import MemoryInterface
from pyrit.models import PromptRequestResponse
from pyrit.models.prompt_request_piece import PromptRequestPiece
from pyrit.prompt_target import PromptTarget
from pyrit.prompt_target.prompt_chat_target.openai_chat_target import AzureOpenAIChatTarget
from pyrit.prompt_normalizer import data_serializer_factory

logger = logging.getLogger(__name__)


class ImageSizing(Enum):
    SIZE256 = "256x256"
    SIZE512 = "512x512"
    SIZE1024 = "1024x1024"


class SupportedDalleVersions(Enum):
    V2 = "dall-e-2"
    V3 = "dall-e-3"


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
        num_images (int, optional): The num ber of output images to generate.
            Defaults to 1. For DALLE-3, can only be 1, for DALLE-2 max is 10 images.
        dalle_version (int, optional): Version of DALLE service. Defaults to 3.
        memory:
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
        image_size: ImageSizing = ImageSizing.SIZE1024,
        num_images: int = 1,
        dalle_version: SupportedDalleVersions = SupportedDalleVersions.V2,
        memory: MemoryInterface | None = None,
        headers: dict = None,
        quality: str = "standard",
        style: str = "natural",
    ):

        super().__init__(memory=memory)

        # make sure number of images is allowed by Dall-e version
        if dalle_version == SupportedDalleVersions.V3:
            self.dalle_version = "dall-e-3"
            if num_images != 1:
                raise ValueError("DALL-E-3 can only generate 1 image at a time.")
            if quality == "hd" or quality == "standard":
                self.quality = quality
            else:
                self.quality = "standard"
            if style == "natural" or style == "vivid":
                self.style = style
        elif dalle_version == SupportedDalleVersions.V2:
            self.dalle_version = "dall-e-2"
            if num_images < 1 or num_images > 10:
                raise ValueError("DALL-E-2 can generate only up to 10 images at a time.")

        self.image_size = image_size.value
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

        resp = await self._generate_images_async(prompt=request.converted_prompt_text)
        return self._parse_response_and_add_to_memory(resp, request)

    def _parse_response_and_add_to_memory(
        self, resp: dict, prompt_request: PromptRequestPiece
    ) -> PromptRequestResponse:
        if "error" not in resp.keys():
            # TODO: fix this to get rid of output filename

            data = data_serializer_factory(data_type="image_path")
            b64_data = resp["data"][0]["b64_json"]
            data.save_b64_image(data=b64_data)
            return self._memory.add_response_entries_to_memory(
                request=prompt_request,
                response_text_pieces=[data.prompt_text],
                response_type="image_path",
                prompt_metadata=json.dumps(resp),
            )

        else:
            parsed_resp = {}
            if resp["exception type"] == "Blocked":

                # Parsing Error Response to form json object
                parsed_resp["exception type"] = "Blocked"
                parsed_resp["data"] = ""
                error_message = str(resp["error"]).split("{")
                parsed_error = "{".join(error_message[1:])

                return self._memory.add_response_entries_to_memory(
                    request=prompt_request,
                    response_text_pieces=[],
                    response_type="image_path",
                    prompt_metadata=json.dumps("{" + parsed_error),
                    error="blocked",
                )

            elif resp["exception type"] == "JSON Error":
                logger.error(f"Response could not be interpreted in the JSON format\n{resp['error']}")
                raise
            else:
                logger.error(f"Error in calling deployment {self.deployment_name}\n{resp['error']}")
                raise

    async def _generate_images_async(self, prompt: str) -> dict:
        try:
            if self.dalle_version == "dall-e-3":
                response = await self.image_target._async_client.images.generate(
                    model=self.deployment_name,
                    prompt=prompt,
                    n=self.n,
                    size=self.image_size,  # type: ignore
                    response_format="b64_json",
                    quality=self.quality,  # type: ignore
                    style=self.style,  # type: ignore
                )
            else:
                response = await self.image_target._async_client.images.generate(
                    model=self.deployment_name,
                    prompt=prompt,
                    n=self.n,
                    size=self.image_size,  # type: ignore
                    response_format="b64_json",
                )
            json_response = json.loads(response.model_dump_json())
            return json_response
        except BadRequestError as e:
            return {"error": e, "exception type": "Blocked"}
        except json.JSONDecodeError as e:
            return {"error": e, "exception type": "JSON Error"}
        except Exception as e:
            return {"error": e, "exception type": "exception"}

    def validate_request(self, *, prompt_request: PromptRequestResponse) -> None:
        if len(prompt_request.request_pieces) != 1:
            raise ValueError("This target only supports a single prompt request piece.")

        if prompt_request.request_pieces[0].converted_prompt_data_type != "text":
            raise ValueError("This target only supports text prompt input.")

        request = prompt_request.request_pieces[0]
        messages = self._memory.get_chat_messages_with_conversation_id(conversation_id=request.conversation_id)

        if len(messages) > 0:
            raise ValueError("This target only supports a single turn conversation.")
