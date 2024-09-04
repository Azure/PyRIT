# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
import httpx  

from pyrit.prompt_target.prompt_target import PromptTarget
from pyrit.common.prompt_template_generator import PromptTemplateGenerator
from pyrit.memory import MemoryInterface
from pyrit.models.prompt_request_response import PromptRequestResponse, construct_response_from_request
from pyrit.models import ChatMessage


logger = logging.getLogger(__name__)


class HuggingFaceEndpointTarget(PromptTarget):
    """The HuggingFaceEndpointTarget interacts with HuggingFace models hosted on cloud endpoints.
    Inherits from PromptTarget to comply with the current design standards.
    """

    def __init__(
        self,
        *,
        api_key: str,
        endpoint: str,
        model_id: str,
        max_tokens: int = 400,
        temperature: float = 1.0,
        top_p: float = 1.0,
        memory: MemoryInterface = None,
        verbose: bool = False,
    ) -> None:
        super().__init__(memory=memory, verbose=verbose)
        self.api_key = api_key
        self.endpoint = endpoint
        self.model_id = model_id
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p

        # Initialize the PromptTemplateGenerator
        self.prompt_template_generator = PromptTemplateGenerator()

    async def send_prompt_async(self, *, prompt_request: PromptRequestResponse) -> PromptRequestResponse:
        """
        Sends a normalized prompt asynchronously to a cloud-based HuggingFace model endpoint.
        """
        self._validate_request(prompt_request=prompt_request)
        request = prompt_request.request_pieces[0]
        headers = {"Authorization": f"Bearer {self.api_key}"}
        payload = {
            "inputs": request.converted_value,
            "parameters": {
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
                "top_p": self.top_p,
            },
        }

        logger.info(f"Sending the following prompt to the cloud endpoint: {request.converted_value}")

        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(self.endpoint, headers=headers, json=payload)
                response_data = response.json()

                if response.status_code != 200:
                    logger.error(f"Failed to get a response from Hugging Face API: {response_data}")
                    raise ValueError(f"Error from Hugging Face API: {response_data}")

                # Check if the response is a list and handle appropriately
                if isinstance(response_data, list):
                    # Access the first element if it's a list and extract 'generated_text' safely
                    response_message = response_data[0].get('generated_text', '')
                else:
                    response_message = response_data.get('generated_text', '')

                prompt_response = construct_response_from_request(
                    request=request,
                    response_text_pieces=[response_message],
                    prompt_metadata={"model_id": self.model_id},
                )
                return prompt_response

            except Exception as e:
                logger.error(f"Error occurred during HTTP request to the Hugging Face endpoint: {e}")
                raise

    async def complete_chat(
        self,
        messages: list[ChatMessage],
        max_tokens: int = 400,
        temperature: float = 1.0,
        top_p: int = 0.9,
    ) -> str:
        """
        Completes a chat interaction by sending a request to a cloud-based HuggingFace model endpoint.
        """
        prompt_template = self.prompt_template_generator.generate_template(messages)
        headers = {"Authorization": f"Bearer {self.api_key}"}
        payload = {
            "inputs": prompt_template,
            "parameters": {
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
            },
        }

        logger.info(f"Sending the following chat message to the cloud endpoint: {prompt_template}")

        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(self.endpoint, headers=headers, json=payload)
                response_data = response.json()

                if response.status_code != 200:
                    logger.error(f"Failed to get a response from Hugging Face API: {response_data}")
                    raise ValueError(f"Error from Hugging Face API: {response_data}")

                # Check if the response is a list and handle appropriately
                if isinstance(response_data, list):
                    response_message = response_data[0].get('generated_text', '')
                else:
                    response_message = response_data.get('generated_text', '')

                return response_message

            except Exception as e:
                logger.error(f"Error occurred during HTTP request to the Hugging Face endpoint: {e}")
                raise


    def _validate_request(self, *, prompt_request: PromptRequestResponse) -> None:
        """
        Validates the provided prompt request response.
        """
        if len(prompt_request.request_pieces) != 1:
            raise ValueError("This target only supports a single prompt request piece.")

        if prompt_request.request_pieces[0].converted_value_data_type != "text":
            raise ValueError("This target only supports text prompt input.")
