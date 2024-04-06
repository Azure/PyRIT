# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import json
import logging
import os
import pathlib
from enum import Enum
import uuid

import requests
from openai import BadRequestError

from pyrit.common.path import RESULTS_PATH
from pyrit.memory.memory_models import PromptRequestResponse
from pyrit.prompt_target import PromptTarget
from pyrit.prompt_target.prompt_chat_target.openai_chat_target import AzureOpenAIChatTarget

logger = logging.getLogger(__name__)


class ImageSizing(Enum):
    SIZE256 = "256x256"
    SIZE512 = "512x512"
    SIZE1024 = "1024x1024"


class ResponseFormat(Enum):
    B64 = "b64_json"
    URL = "url"


class SupportedDalleVersions(Enum):
    V2 = 2
    V3 = 3


class ImageTarget(PromptTarget):
    """
    The ImageTarget takes a prompt and generates images
    This class initializes a DALL-E image target

    Args:
        deployment_name (str): The name of the deployment.
        endpoint (str): The endpoint URL for the service.
        api_key (str): The API key for accessing the service.
        api_version (str, optional): The API version. Defaults to "2024-02-01".
        image_size (str, optional): The size of the image to output, must be a value of VALID_SIZES.
            Defaults to 1024x1024.
        response_format (str, optional): Format of output (base64 or url). Defaults to b64_json.
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
        response_format: ResponseFormat | None = ResponseFormat.B64,
        num_images: int | None = 1,
        dalle_version: SupportedDalleVersions | None = SupportedDalleVersions.V2,
    ):
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
        self.response_format = response_format.value
        self.n = num_images

        self.deployment_name = deployment_name
        self.image_target = AzureOpenAIChatTarget(
            deployment_name=deployment_name, endpoint=endpoint, api_key=api_key, api_version=api_version
        )
        self.output_dir = pathlib.Path(RESULTS_PATH) / "images"

    def download_image(self, image_url: str, output_filename: str) -> str:
        """
        Downloads the image from a URL and stores the image locally
        Args:
            image_url (str): URL which image is stored at
            output_filename (str): name of file to store image in
        Returns: file location
        """

        # This will likely be replaced once our memory is refactored!
        # If the directory doesn't exist, create it
        if not os.path.isdir(self.output_dir):
            os.mkdir(self.output_dir)

        # Initialize the image path
        image_path = self.output_dir / output_filename
        # Retrieve the generated image

        generated_image = requests.get(image_url).content  # download the image
        with open(image_path, "wb") as image_file:
            image_file.write(generated_image)
        return str(image_path)

    def send_prompt(
        self,
        *,
        prompt_request: PromptRequestResponse,
    ) -> PromptRequestResponse:
        """
        Sends prompt to image target and returns response
        Args:
            normalized_prompt (str): the prompt to send
            conversation_id (str): the ID of the conversation for memory
            normalizer_id (str): the ID provided by the prompt normalizer
        Returns: response from target model in a JSON format
        """

        output_filename = f"{uuid.uuid4()}.png"

        self._memory.insert_prompt_entries(entries=prompt_request.request_pieces)

        resp = self._generate_images(prompt=prompt_request.request_pieces[0].converted_prompt_text)

        if "error" not in resp.keys():
            if self.response_format == "url":
                image_url = resp["data"][0]["url"]  # extract image URL from response
                image_location = self.download_image(image_url=image_url, output_filename=output_filename)
                resp["image_file_location"] = image_location  # append where stored image locally to response
            return json.dumps(resp)
        else:
            if resp["exception type"] == "Blocked":
                logger.error(f"Content Blocked\n{resp['error']}")
                return ""
            elif resp["exception type"] == "JSON Error":
                logger.error(f"Response could not be interpreted in the JSON format\n{resp['error']}")
                raise
            else:
                logger.error(f"Error in calling deployment {self.deployment_name}\n{resp['error']}")
                raise

    async def send_prompt_async(self, *, normalized_prompt: str, conversation_id: str, normalizer_id: str) -> str:
        """
        (Async) Sends prompt to image target and returns response
        Args:
            normalized_prompt (str): the prompt to send
            conversation_id (str): the ID of the conversation for memory
            normalizer_id (str): the ID provided by the prompt normalizer
        Returns: response from target model in a JSON format
        """

        output_filename = f"{conversation_id}.png"
        resp = await self._generate_images_async(prompt=normalized_prompt)

        # TODO add response to memory

        if "error" not in resp:
            if self.response_format == "url":
                image_url = resp["data"][0]["url"]  # extract image URL from response
                image_location = self.download_image(image_url=image_url, output_filename=output_filename)
                resp["image_file_location"] = image_location  # append where stored image locally to response
            return json.dumps(resp)
        else:
            if resp["exception type"] == "Blocked":
                logger.error(f"Content Blocked\n{resp['error']}")
                return ""
            elif resp["exception type"] == "JSON Error":
                logger.error(f"Response could not be interpreted in the JSON format\n{resp['error']}")
                raise
            else:
                logger.error(f"Error in calling deployment {self.deployment_name}\n{resp['error']}")
                raise

    async def _generate_images_async(self, prompt: str) -> dict:
        try:
            response = await self.image_target._async_client.images.generate(
                model=self.deployment_name,
                prompt=prompt,
                n=self.n,
                size=self.image_size,
                response_format=self.response_format,
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
                response_format=self.response_format,
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
