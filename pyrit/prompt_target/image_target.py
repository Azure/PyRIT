# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

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
        
    def send_prompt(self, prompt: str, conversation_id: str, normalizer_id: str, output_filename: str="image1.png"):

        resp = self.image_target.complete_image_chat(prompt=prompt, num_images=self.n, output_filename=output_filename)
        return resp
    
    def send_prompt_async(): #TODO
        return