# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import asyncio
import logging
import time

import requests

from pyrit.common.constants import MAX_RETRY_API_COUNT
from pyrit.common.net import HttpClientSession
from pyrit.common.prompt_template_generator import PromptTemplateGenerator
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

    def __init__(
        self,
        *,
        endpoint_uri: str,
        api_key: str,
    ) -> None:
        """
        Args:
            endpoint_uri: AML online endpoint URI.
            api_key: api key for the endpoint
        """
        # AML online endpoint details
        self.endpoint_uri: str = endpoint_uri
        self.api_key: str = api_key

        self.prompt_template_generator = PromptTemplateGenerator()

    def get_headers(self) -> dict:
        """Headers for accessing inference endpoint deployed in AML.
        Returns:
            headers(dict): contains bearer token as aml key and content-type: JSON
        """
        if self.api_key == "":
            raise ValueError(
                "AML Managed Online Endpoint 'api_key' value is empty, please provide a valid endpoint 'api_key' value."
            )

        headers: dict = {
            "Content-Type": "application/json",
            "Authorization": ("Bearer " + self.api_key),
        }

        return headers

    def construct_payload(self, prompt_template: str, max_tokens: int, temperature: float, top_p: int) -> dict:
        """Constructs a payload in the format required by the endpoint.

        Args:
            prompt_template (str): The template string for the prompt to be sent to the endpoint.
            max_tokens (int): The maximum number of tokens to be used in the response.
            temperature (float): The temperature setting for the model creativity in generating responses.
            top_p (int): The top_p setting controlling the diversity of the response.

        Returns:
            dict: A payload dictionary formatted as per the endpoint's requirements.
        """
        data = {
            "input_data": {
                "inputs": {"input_string": [prompt_template]},
                "parameters": {
                    "max_new_tokens": max_tokens,
                    "temperature": temperature,
                    "top_p": top_p,
                },
            }
        }
        return data

    async def _generate_and_extract_response(
        self,
        messages: list[ChatMessage],
        max_tokens: int,
        temperature: float,
        top_p: int,
        is_async: bool = False,
    ) -> str:
        """
        Generates and extracts response from the AML endpoint.

        Args:
            messages (list[ChatMessage]): The chat messages objects containing the role and content.
            max_tokens (int): The maximum number of tokens to generate.
            temperature (float): Controls randomness in the response generation.
            top_p (int): Controls diversity of the response generation.
            is_async (bool): Flag to determine if the request is asynchronous.

        Returns:
            str: The generated response message.
        """
        prompt_template = self.prompt_template_generator.generate_template(messages)
        headers = self.get_headers()
        payload = self.construct_payload(prompt_template, max_tokens, temperature, top_p)

        if is_async:
            response = await self._send_async_request(headers, payload)
        else:
            response = self._send_sync_request(headers, payload)
        return self.extract_first_response_message(response)

    def extract_first_response_message(self, response_message: list[dict[str, str]]) -> str:
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

    async def _send_async_request(self, headers: dict, payload: dict) -> list[dict[str, str]]:
        """
        Sends an asynchronous request to the AML endpoint.

        Args:
            headers(dict): contains bearer token as aml key and content-type: JSON
            payload (dict): Request payload.

        Returns:
            str: The generated response message.
        """
        generated_response: list[dict[str, str]] = None
        try:
            http_session = HttpClientSession.get_client_session()
            curr_retry_count = 0
            while True:
                async with http_session.post(url=self.endpoint_uri, json=payload, headers=headers) as response:
                    if response.status == 429 or response.status == 408:
                        await asyncio.sleep(10)
                        continue

                    if response.status >= 500 and curr_retry_count < MAX_RETRY_API_COUNT:
                        curr_retry_count += 1
                        await asyncio.sleep(10)
                        continue

                    if response.status >= 400:
                        text = await response.text()
                        raise RuntimeError(f"HTTP error: {response.status}\n{text}")
                generated_response = await response.json()
                break
            return generated_response
        except Exception as e:
            logger.error(f"Error occured during inference: {e}")
            raise

    def _send_sync_request(self, headers: dict, payload: dict) -> list[dict[str, str]]:
        """
        Sends a synchronous request to the AML endpoint.

        Args:
            headers (dict): Request headers.
            payload (dict): Request payload.

        Returns:
            list[dict[str, str]]: The response from the endpoint.
        """
        try:
            curr_retry_count = 0
            while True:
                response = requests.post(self.endpoint_uri, json=payload, headers=headers)
                if response.status_code == 429 or response.status_code == 408:
                    # error codes, 429 refers 'too many requests' and 408 refers 'request timeout'
                    time.sleep(10)
                    continue

                if response.status_code >= 500 and curr_retry_count < MAX_RETRY_API_COUNT:
                    curr_retry_count += 1
                    time.sleep(10)
                    continue

                if response.status_code >= 400:
                    error_message = response.text
                    raise RuntimeError(f"HTTP error: {response.status_code}\n{error_message}.")

                generated_response = response.json()
                break
            return generated_response
        except Exception as e:
            logger.error(f"Error occured during inference: {e}")
            raise

    async def complete_chat_async(
        self,
        messages: list[ChatMessage],
        max_tokens: int = 400,
        temperature: float = 1.0,
        top_p: int = 1,
    ) -> str:
        """Async method that completes a chat interaction by generating a response to the given input prompt.

        Args:
            message (ChatMessage): The chat message object containing the role and content.
            max_tokens (int, optional): The maximum number of tokens to generate. Defaults to 512.
            temperature (float, optional): Controls randomness in the response generation. Defaults to 1.0.
            top_p (int, optional): Controls diversity of the response generation. Defaults to 1.

        Raises:
            Exception: For any errors during the process.

        Returns:
            str: The generated response message.
        """
        return await self._generate_and_extract_response(messages, max_tokens, temperature, top_p, is_async=True)

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
        # Run the asynchronous function _generate_and_extract_response method synchronously
        response = _loop.run_until_complete(
            self._generate_and_extract_response(messages, max_tokens, temperature, top_p, is_async=False)
        )
        return response
