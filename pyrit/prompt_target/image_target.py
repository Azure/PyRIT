# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import json
import requests
import os
import pathlib

from pyrit.common.path import RESULTS_PATH
from pyrit.prompt_target import PromptTarget
from pyrit.prompt_target.azure_openai_chat_target import AzureOpenAIChatTarget

class ImageTarget(PromptTarget):
    """
    The ImageTarget takes prompt and generates images
    """
    def __init__(self, deployment_name: str = None, endpoint: str = None, 
                 api_key: str = None, api_version: str = "2024-02-01", num_images: int = 1,
                 temperature: float = 1.0):
        
        # make sure number of images is in a reasonable range
        if num_images < 1 or num_images > 100:
            num_images = 1

        self.n = num_images

        self.temperature = temperature
        self.deployment_name = deployment_name
        self.image_target = AzureOpenAIChatTarget(
            deployment_name=deployment_name, endpoint=endpoint, api_key=api_key, api_version=api_version
        )
        self.output_dir = pathlib.Path(RESULTS_PATH) / "images"

    def dowhload_image(self, image_json: json, output_filename: str):
        """
        Parses the JSON response to get the URL and downloads the image from that URL and stores image locally 
        Parameters:
            image_json: response from image target in JSON format
            output_filename: (optional) name of file to store image in
        Returns: file location
        """

        # If the directory doesn't exist, create it
        if not os.path.isdir(self.output_dir):
            os.mkdir(self.output_dir)

        # Initialize the image path (note the filetype should be png)
        image_path = self.output_dir / output_filename
        # Retrieve the generated image
        image_url = image_json["data"][0]["url"]  # extract image URL from response
        generated_image = requests.get(image_url).content  # download the image
        with open(image_path, "wb") as image_file:
            image_file.write(generated_image)
        return image_path
    
    def send_prompt(self, prompt: str, conversation_id: str = None, normalizer_id: str = None):
        """
        Sends prompt to image target and returns response
        Parameters:
            prompt: a string with the prompt to send
        Returns: response from target model in a JSON format
        """
        output_filename = conversation_id + "_" + normalizer_id + ".png" # name of file based on conversation ID and normalizer ID
        resp = self.image_target.complete_image_chat(prompt=prompt, num_images=self.n)
        image_location = self.dowhload_image(image_json = resp, output_filename=output_filename)
        resp["image_file_location"] = image_location
        return resp
    
    def send_prompt_async(): #TODO
        return