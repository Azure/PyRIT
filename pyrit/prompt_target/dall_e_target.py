# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import json
import logging
import pathlib
from enum import Enum
import uuid

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
    V2 = 2
    V3 = 3


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
    """

    def __init__(
        self,
        *,
        deployment_name: str = None,
        endpoint: str = None,
        api_key: str = None,
        api_version: str | None = "2024-02-01",
        image_size: ImageSizing | None = ImageSizing.SIZE1024,
        num_images: int | None = 1,
        dalle_version: SupportedDalleVersions | None = SupportedDalleVersions.V2,
        memory: MemoryInterface | None = None,
    ):

        super().__init__(memory=memory)

        if num_images is None:
            num_images = 1  # set 1 as default

        # make sure number of images is allowed by Dall-e version
        if dalle_version == SupportedDalleVersions.V3:
            if num_images != 1:
                raise ValueError("DALL-E-3 can only generate 1 image at a time.")
        elif dalle_version == SupportedDalleVersions.V2:
            if num_images < 1 or num_images > 10:
                raise ValueError("DALL-E-2 can generate only up to 10 images at a time.")

        self.image_size = image_size.value
        self.n = num_images

        self.deployment_name = deployment_name
        self.image_target = AzureOpenAIChatTarget(
            deployment_name=deployment_name, endpoint=endpoint, api_key=api_key, api_version=api_version
        )
        self.output_dir = pathlib.Path(RESULTS_PATH) / "images"

    def send_prompt(
        self,
        *,
        prompt_request: PromptRequestResponse,
    ) -> PromptRequestResponse:
        """
        Sends prompt to image target and returns response
        Args:
            prompt_request (PromptRequestResponse): the prompt to send formatted as an object
        Returns: response from target model formatted as an object
        """

        request = prompt_request.request_pieces[0]

        output_filename = f"{uuid.uuid4()}.png"

        self._memory.add_request_pieces_to_memory(request_pieces=[request])

        resp = self._generate_images(prompt=request.converted_prompt_text)

        return self._parse_response_and_add_to_memory(resp, output_filename, request)

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

        request = prompt_request.request_pieces[0]

        output_filename = f"{uuid.uuid4()}.png"

        self._memory.add_request_pieces_to_memory(request_pieces=[request])

        resp = await self._generate_images_async(prompt=request.converted_prompt_text)
        return self._parse_response_and_add_to_memory(resp, output_filename, request)

    def _parse_response_and_add_to_memory(
        self, resp: dict, output_filename: str, prompt_request: PromptRequestPiece
    ) -> PromptRequestResponse:
        if "error" not in resp.keys():
            image_location = output_filename
            data = data_serializer_factory(data_type="image_path")
            b64_data = resp["data"][0]["b64_json"]
            data.save_b64_image(data=b64_data, output_filename=image_location)
            return self._memory.add_response_entries_to_memory(
                request=prompt_request,
                response_text_pieces=[image_location],
                response_type="image_path",
                prompt_metadata=json.dumps(resp),
            )

        else:
            if resp["exception type"] == "Blocked":

                return self._memory.add_response_entries_to_memory(
                    request=prompt_request,
                    response_text_pieces=[],
                    response_type="image_path",
                    prompt_metadata=json.dumps(resp),
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
            response = await self.image_target._async_client.images.generate(
                model=self.deployment_name, prompt=prompt, n=self.n, size=self.image_size, response_format="b64_json"
            )
            json_response = json.loads(response.model_dump_json())
            return json_response
        except BadRequestError as e:
            return {"error": e, "exception type": "Blocked"}
        except json.JSONDecodeError as e:
            return {"error": e, "exception type": "JSON Error"}
        except Exception as e:
            return {"error": e, "exception type": "exception"}

    def _generate_images(self, prompt: str) -> dict:
        try:
            response = self.image_target._client.images.generate(
                model=self.deployment_name,
                prompt=prompt,
                n=self.n,
                size=self.image_size,
                response_format="b64_json",
            )
            json_response = json.loads(response.model_dump_json())
            return json_response
        except BadRequestError as e:
            logger.error(f"Content Blocked\n{e}")
            return {"error": e}
        except json.JSONDecodeError as e:
            logger.error(f"Response could not be interpreted in the JSON format\n{e}")
        except Exception as e:
            logger.error(f"Error in calling deployment {self.deployment_name}\n{e}")
        return {}
