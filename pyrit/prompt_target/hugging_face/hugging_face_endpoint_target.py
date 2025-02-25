# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging

from pyrit.common.net_utility import make_request_and_raise_if_error_async
from pyrit.models.prompt_request_response import (
    PromptRequestResponse,
    construct_response_from_request,
)
from pyrit.prompt_target import PromptTarget

logger = logging.getLogger(__name__)


class HuggingFaceEndpointTarget(PromptTarget):
    """The HuggingFaceEndpointTarget interacts with HuggingFace models hosted on cloud endpoints.

    Inherits from PromptTarget to comply with the current design standards.
    """

    def __init__(
        self,
        *,
        hf_token: str,
        endpoint: str,
        model_id: str,
        max_tokens: int = 400,
        temperature: float = 1.0,
        top_p: float = 1.0,
        verbose: bool = False,
    ) -> None:
        """Initializes the HuggingFaceEndpointTarget with API credentials and model parameters.

        Args:
            hf_token (str): The Hugging Face token for authenticating with the Hugging Face endpoint.
            endpoint (str): The endpoint URL for the Hugging Face model.
            model_id (str): The model ID to be used at the endpoint.
            max_tokens (int, Optional): The maximum number of tokens to generate. Defaults to 400.
            temperature (float, Optional): The sampling temperature to use. Defaults to 1.0.
            top_p (float, Optional): The cumulative probability for nucleus sampling. Defaults to 1.0.
            verbose (bool, Optional): Flag to enable verbose logging. Defaults to False.
        """
        super().__init__(verbose=verbose)
        self.hf_token = hf_token
        self.endpoint = endpoint
        self.model_id = model_id
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p

    async def send_prompt_async(self, *, prompt_request: PromptRequestResponse) -> PromptRequestResponse:
        """
        Sends a normalized prompt asynchronously to a cloud-based HuggingFace model endpoint.

        Args:
            prompt_request (PromptRequestResponse): The prompt request containing the input data and associated details
            such as conversation ID and role.

        Returns:
            PromptRequestResponse: A response object containing generated text pieces as a list of `PromptRequestPiece`
                objects. Each `PromptRequestPiece` includes the generated text and relevant information such as
                conversation ID, role, and any additional response attributes.

        Raises:
            ValueError: If the response from the Hugging Face API is not successful.
            Exception: If an error occurs during the HTTP request to the Hugging Face endpoint.
        """
        self._validate_request(prompt_request=prompt_request)
        request = prompt_request.request_pieces[0]
        headers = {"Authorization": f"Bearer {self.hf_token}"}
        payload: dict[str, object] = {
            "inputs": request.converted_value,
            "parameters": {
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
                "top_p": self.top_p,
            },
        }

        logger.info(f"Sending the following prompt to the cloud endpoint: {request.converted_value}")

        try:
            # Use the utility method to make the request
            response = await make_request_and_raise_if_error_async(
                endpoint_uri=self.endpoint,
                method="POST",
                request_body=payload,
                headers=headers,
                post_type="json",
            )

            response_data = response.json()

            # Check if the response is a list and handle appropriately
            if isinstance(response_data, list):
                # Access the first element if it's a list and extract 'generated_text' safely
                response_message = response_data[0].get("generated_text", "")
            else:
                response_message = response_data.get("generated_text", "")

            prompt_response = construct_response_from_request(
                request=request,
                response_text_pieces=[response_message],
                prompt_metadata={"model_id": self.model_id},
            )
            return prompt_response

        except Exception as e:
            logger.error(f"Error occurred during HTTP request to the Hugging Face endpoint: {e}")
            raise

    def _validate_request(self, *, prompt_request: PromptRequestResponse) -> None:
        """
        Validates the provided prompt request response.

        Args:
            prompt_request (PromptRequestResponse): The prompt request to validate.

        Raises:
            ValueError: If the request is not valid for this target.
        """
        if len(prompt_request.request_pieces) != 1:
            raise ValueError("This target only supports a single prompt request piece.")

        if prompt_request.request_pieces[0].converted_value_data_type != "text":
            raise ValueError("This target only supports text prompt input.")

    def is_json_response_supported(self) -> bool:
        """Indicates that this target supports JSON response format."""
        return False
