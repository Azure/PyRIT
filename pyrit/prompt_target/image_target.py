# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import json
import requests
import os
import pathlib
import logging

from openai import BadRequestError
from enum import Enum
from pyrit.common.path import RESULTS_PATH
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
    # The ImageTarget takes prompt and generates images
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
        """
        Class that initializes a DALLE image target

        Args:
            deployment_name (str): The name of the deployment.
            endpoint (str): The endpoint URL for the service.
            api_key (str): The API key for accessing the service.
            api_version (str, optional): The API version. Defaults to "2024-02-01".
            image_size (str, optional): The size of the image to output, must be a value of VALID_SIZES.
                Defaults to 1024x1024.
            response_format (str, optional): Format of output (base64 or url). Defaults to b64_json.
            num_images (int, optional): The number of output images to generate.
                Defaults to 1. For DALLE-3, can only be 1, for DALLE-2 max is 10 images.
            dalle_version (int, optional): Version of DALLE service. Defaults to 3.

        """

        # make sure number of images is allowed by Dalle version
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
        Parameters:
            image_url: string with URL which image is stored at
            output_filename: name of file to store image in
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
        return image_path

    def send_prompt(
        self, normalized_prompt: str, conversation_id: str | None = None, normalizer_id: str | None = None
    ) -> dict:
        """
        Sends prompt to image target and returns response
        Parameters:
            prompt: a string with the prompt to send
        Returns: response from target model in a JSON format
        """
        output_filename = f"{conversation_id}_{normalizer_id}.png"
        resp = self.generate_images(prompt=normalized_prompt)
        if resp:
            if self.response_format == "url":
                image_url = resp["data"][0]["url"]  # extract image URL from response
                image_location = self.download_image(image_url=image_url, output_filename=output_filename)
                resp["image_file_location"] = image_location  # append where stored image locally to response
            return resp
        else:
            return None

    async def send_prompt_async(self, *, normalized_prompt: str, conversation_id: str, normalizer_id: str) -> dict:
        output_filename = f"{conversation_id}_{normalizer_id}.png"
        resp = self.generate_images(prompt=normalized_prompt)
        if resp:
            if self.response_format == "url":
                image_url = resp["data"][0]["url"]  # extract image URL from response
                image_location = self.download_image(image_url=image_url, output_filename=output_filename)
                resp["image_file_location"] = image_location  # append where stored image locally to response
            return resp
        else:
            return None

    async def generate_images_async(self, prompt:str):
        try:
            response = await self.image_target._client.images.generate(
                model=self.deployment_name,
                prompt=prompt,
                n=self.n,
                size=self.image_size,
                response_format=self.response_format,
            )
        except BadRequestError as e:
            print(e)
            logger.error("Content Blocked\n" + str(e))
            return None
        except Exception as e:
            logger.error("API Call Error\n" + str(e))
            return None
        try:
            json_response = json.loads(response.model_dump_json())
        except json.JSONDecodeError:
            logger.error("Response could not be interpreted in the JSON format")
            logger.error(json_response)
            return None
        return json_response
    def generate_images(self, prompt: str) -> dict:
        """
        Sends prompt to image target and returns response
        Parameters:
            prompt: a string with the prompt to send
        Returns: response from target model in a JSON format
        """
        try:
            response = self.image_target._client.images.generate(
                model=self.deployment_name,
                prompt=prompt,
                n=self.n,
                size=self.image_size,
                response_format=self.response_format,
            )
        except BadRequestError as e:
            print(e)
            logger.error("Content Blocked\n" + str(e))
            return None
        except Exception as e:
            logger.error("API Call Error\n" + str(e))
            return None
        try:
            json_response = json.loads(response.model_dump_json())
        except json.JSONDecodeError:
            logger.error("Response could not be interpreted in the JSON format")
            logger.error(json_response)
            return None
        return json_response
