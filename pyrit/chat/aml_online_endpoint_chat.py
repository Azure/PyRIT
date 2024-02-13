# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import asyncio
import logging

from pyrit.common import environment_variables, net_utility
from pyrit.interfaces import ChatSupport
from pyrit.models import ChatMessage

logger = logging.getLogger(__name__)
_loop = asyncio.get_event_loop()


class AMLOnlineEndpointChat(ChatSupport):
    """The AMLOnlineEndpointChat interacts with AML-managed online endpoints, specifically
    for conducting red teaming activities.

    Args:
        ChatSupport (abc.ABC): Implementing methods for interactions with the AML endpoint
    """

    API_KEY_ENVIRONMENT_VARIABLE: str = "AZURE_ML_API_KEY"
    ENDPOINT_URI_ENVIRONMENT_VARIABLE: str = "AZURE_ML_MANAGED_ENDPOINT"

    def __init__(
        self,
        *,
        endpoint_uri: str = None,
        api_key: str = None,
    ) -> None:
        """
        Args:
            endpoint_uri: AML online endpoint URI.
            api_key: api key for the endpoint
        """
        self.endpoint_uri: str = environment_variables.get_required_value(
            self.ENDPOINT_URI_ENVIRONMENT_VARIABLE, endpoint_uri
        )
        self.api_key: str = environment_variables.get_required_value(self.API_KEY_ENVIRONMENT_VARIABLE, api_key)

    def complete_chat(
        self,
        messages: list[ChatMessage],
        max_tokens: int = 400,
        temperature: float = 1.0,
        top_p: int = 1,
    ) -> str:
        """Completes a chat interaction by generating a response to the given input prompt.
        This is a synchronous wrapper for the asynchronous _generate_and_extract_response method.

        Args:
            messages (list[ChatMessage]): The chat messages objects containing the role and content.
            max_tokens (int, optional): The maximum number of tokens to generate. Defaults to 400.
            temperature (float, optional): Controls randomness in the response generation. Defaults to 1.0.
            top_p (int, optional): Controls diversity of the response generation. Defaults to 1.

        Raises:
            Exception: For any errors during the process.

        Returns:
            str: The generated response message.
        """

        headers = self._get_headers()
        payload = self._construct_http_body(messages, max_tokens, temperature, top_p)

        response = net_utility.make_request_and_raise_if_error(
            self.endpoint_uri, method="POST", request_body=payload, headers=headers
        )
        return response.json()

    def _construct_http_body(
        self, messages: list[ChatMessage], max_tokens: int, temperature: float, top_p: int
    ) -> dict:
        """Constructs a http body in the format required by the endpoint.

        Args:
            prompt_template (str): The template string for the prompt to be sent to the endpoint.
            max_tokens (int): The maximum number of tokens to be used in the response.
            temperature (float): The temperature setting for the model creativity in generating responses.
            top_p (int): The top_p setting controlling the diversity of the response.

        Returns:
            dict: A payload dictionary formatted as per the endpoint's requirements.
        """
        messages_dict = [message.dict() for message in messages]

        data = {
            "input_data": {
                "input_string": messages_dict,
                "parameters": {
                    "max_new_tokens": max_tokens,
                    "temperature": temperature,
                    "top_p": top_p,
                    "top_k": 50,
                    "stop": ["</s>"],
                    "stop_sequences": ["</s>"],
                    "return_full_text": False,
                    "repetition_penalty": 1.2,
                },
            }
        }
        return data

    def _extract_first_response_message(self, response_message: list[dict[str, str]]) -> str:
        """Extracts the first message from a list of response messages.

        Each message in the list is expected to be a dictionary with a key '0' that holds the message string.
        This function retrieves the message corresponding to the key '0' from the first dictionary in the list.

        Args:
            response_message (list[dict[str, str]]): A list of dictionaries containing response messages.

        Raises:
            ValueError: If the list is empty or the first dictionary does not contain the key '0'.

        Returns:
            str: The first response message extracted from the list.
        """
        if not response_message:
            raise ValueError("The response_message list is empty.")
        first_response_dict = response_message[0]
        if "0" not in first_response_dict:
            raise ValueError(
                f"Key '0' does not exist in the first response message. "
                f"Unable to retrieve first response message from the endpoint {self.endpoint_uri}. "
                f"Response message: {first_response_dict}"
            )
        return first_response_dict["0"]

    def _get_headers(self) -> dict:
        """Headers for accessing inference endpoint deployed in AML.
        Returns:
            headers(dict): contains bearer token as aml key and content-type: JSON
        """

        headers: dict = {
            "Content-Type": "application/json",
            "Authorization": ("Bearer " + self.api_key),
        }

        return headers
