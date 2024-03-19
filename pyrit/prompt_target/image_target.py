# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

#from PIL import Image

from pyrit.prompt_target import PromptTarget
from pyrit.chat import AzureOpenAIChat

class ImageTarget(AzureOpenAIChat, PromptTarget):
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

        AzureOpenAIChat.__init__(
            self, deployment_name=deployment_name, endpoint=endpoint, api_key=api_key, api_version=api_version
        )

    def _prepare_data(self, normalized_prompt: str = None):
        return str.encode(normalized_prompt)
    
    def send_prompt(self, prompt: str, conversation_id: str, normalizer_id: str):
        print("send prompt: ", prompt)
        message = self._prepare_data(prompt)
        print(message)

        resp = self.complete_image_chat(prompt=prompt, num_images=self.n)
        """
        self.memory.add_chat_message_to_memory(
            conversation=ChatMessage(role="user", content=prompt),
            conversation_id=conversation_id, 
            normalizer_id=normalizer_id
        )
        """
        return resp
    
    def send_prompt_async():
        return