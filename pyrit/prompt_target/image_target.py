# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import json
from typing import Any, Coroutine
import requests
import os
import pathlib
import logging

from openai import BadRequestError

from pyrit.common.path import RESULTS_PATH
from pyrit.prompt_target import PromptTarget
from pyrit.prompt_target.openai_chat_target import AzureOpenAIChatTarget

logger = logging.getLogger(__name__)


class ImageTarget(PromptTarget):
    # The ImageTarget takes prompt and generates images
    def __init__(
        self,
        deployment_name: str = None,
        endpoint: str = None,
        api_key: str = None,
        api_version: str = "2024-02-01",
        image_size: str = "1024x1024",
        response_format: str = "b64_json",
        num_images: int = 1,
        dalle_version: int = 3,
    ):
        """
        Class that initializes a DALLE image target

        Args:
            deployment_name (str): The name of the deployment.
            endpoint (str, optional): The endpoint URL for the service.
            api_key (str, optional): The API key for accessing the service.
            api_version (str, optional): The API version. Defaults to "2024-02-01".
            image_size (str, optional): The size of the image to output. Defaults to 1024x1024.
            response_format (str, optional): Format of output. Defaults to b64_json.
            num_images (int, optional): The number of output images to generate.
                Defaults to 1.
            dalle_version (int, optional): Version of DALLE service. Defaults to 3.

        """

        VALID_SIZES = ["256x256", "512x512", "1024x1024"]

        # make sure number of images is allowed by Dalle version
        if dalle_version == 3:
            if num_images != 1:
                raise ValueError("DALL-E-3 can only generate 1 image at a time.")
        elif dalle_version == 2:
            if num_images < 1 or num_images > 10:
                raise ValueError("DALL-E-2 can generate only up to 10 images at a time.")
        else:
            raise ValueError("Unsupported DALL-E version. Only DALL-E 2 and 3 are supported.")

        if image_size not in VALID_SIZES:
            raise ValueError(f"Invalid image size '{image_size}'. Image size must be one of {VALID_SIZES}.")
        self.image_size = image_size

        if response_format != "url" and response_format != "b64_json":
            raise ValueError(f"Invalid response_format '{response_format}'. Must be url or b64_json.")
        self.response_format = response_format

        self.n = num_images

        self.deployment_name = deployment_name
        self.image_target = AzureOpenAIChatTarget(
            deployment_name=deployment_name, endpoint=endpoint, api_key=api_key, api_version=api_version
        )
        self.output_dir = pathlib.Path(RESULTS_PATH) / "images"

    def download_image(self, image_json, output_filename: str):
        """
        Parses the JSON response to get the URL and downloads the image from that URL and stores image locally
        Parameters:
            image_json: response from image target in JSON format
            output_filename: (optional) name of file to store image in
        Returns: file location
        """
        # This will likely be replaced once our memory is refactored!
        # If the directory doesn't exist, create it
        if not os.path.isdir(self.output_dir):
            os.mkdir(self.output_dir)

        # Initialize the image path
        image_path = self.output_dir / output_filename
        # Retrieve the generated image
        image_url = image_json["data"][0]["url"]  # extract image URL from response
        generated_image = requests.get(image_url).content  # download the image
        with open(image_path, "wb") as image_file:
            image_file.write(generated_image)
        return image_path

    def send_prompt(self, normalized_prompt: str, conversation_id: str = None, normalizer_id: str = None):
        """
        Sends prompt to image target and returns response
        Parameters:
            prompt: a string with the prompt to send
        Returns: response from target model in a JSON format
        """
        output_filename = (
            conversation_id + "_" + normalizer_id + ".png"
        )  # name of file based on conversation ID and normalizer ID
        resp = self.complete_image_chat(prompt=normalized_prompt)
        if resp: # This will likely be replaced once our memory is refactored
            if self.response_format == "url": 
                image_location = self.download_image(image_json=resp, output_filename=output_filename)
                resp["image_file_location"] = image_location  # append where stored image locally to response
            return resp
        else:
            return None

    # TODO:
    def send_prompt_async(
        self, *, normalized_prompt: str, conversation_id: str, normalizer_id: str
    ) -> Coroutine[Any, Any, str]:
        return None

    def complete_image_chat(self, prompt: str):
        """
        Sends prompt to image target and returns response
        Parameters:
            prompt: a string with the prompt to send
            num_images: number of images for model to generate
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
            logger.error("Response could not be formatted as a json")
            logger.error(json_response)
            return None
        return json_response
