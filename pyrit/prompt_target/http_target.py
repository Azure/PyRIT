import logging
import json
import re
from typing import Any, Union, Dict
from pyrit.prompt_target import PromptTarget
from pyrit.memory import MemoryInterface
from pyrit.models import construct_response_from_request, PromptRequestPiece, PromptRequestResponse, PromptResponse
from pyrit.common import net_utility

logger = logging.getLogger(__name__)

class HTTPTarget(PromptTarget):
    """
    HTTPTarget is designed to interact with any LLM model that has a REST API. It allows sending and receiving responses
    via HTTP requests. This class supports POST requests, any JSON body, and can handle paths within JSON to send input
    prompts and fetch responses from JSON output given the JSON path.

    Attributes:
        url (str): The URL for the HTTP target.
        response_path (str): The JSON path to fetch the response from the HTTP response body.
        prompt_path (str): The JSON path to place the input prompt in the request body.
        http_headers (Dict[str, str], optional): The HTTP headers for the request. Defaults to None.
        request_body (Dict[str, Any], optional): The request body for the HTTP target. Defaults to None.
        method (str, optional): The HTTP method to use. Defaults to "POST".
        params (Dict[str, str], optional): The query parameters for the HTTP request. Defaults to None.
        memory (Union[MemoryInterface, None], optional): The memory interface for storing conversation history. Defaults to None.
    """

    def __init__(
        self,
        url: str,
        response_path: str,
        prompt_path: str,
        http_headers: Dict[str, str] = None,
        request_body: Dict[str, Any] = None,
        method: str = "POST",
        params: Dict[str, str] = None,
        memory: Union[MemoryInterface, None] = None,
    ) -> None:
        super().__init__(memory=memory)
        self.url = url
        self.http_headers = http_headers
        self.response_path = response_path
        self.body = request_body
        self.method = method
        self.params = params
        self.prompt_path = prompt_path

    async def send_prompt_async(self, *, prompt_request: PromptRequestResponse) -> PromptRequestResponse:
        """
        Sends prompt to HTTP endpoint and returns the response.

        Args:
            prompt_request (PromptRequestResponse): The prompt request to be sent.

        Returns:
            PromptRequestResponse: The response from the HTTP endpoint.
        """
        self._validate_request(prompt_request=prompt_request)
        request = prompt_request.request_pieces[0]

        self.body = self.replace_key(self.body, self.prompt_path, request.converted_value)

        response = await net_utility.make_request_and_raise_if_error_async(
            endpoint_uri=self.url,
            method=self.method,
            request_body=self.body,
            headers=self.http_headers,
            params=self.params
        )

        if response.status_code == 200:
            try:
                resp_text = self.fetch_key(data=json.loads(response.content), key=self.response_path)
            except Exception as e:
                logger.error(e)
                resp_text = str(response.content)  # if HTTP response is not JSON
        else:
            response.raise_for_status()

        prompt_response = PromptResponse(
            completion=resp_text,
            prompt=request.converted_value,
        )
        response_entry = construct_response_from_request(
            request=request,
            response_text_pieces=[prompt_response.completion],
            prompt_metadata=prompt_response.to_json(),
            response_type="text"
        )

        return response_entry

    def fetch_key(self, data: dict, key: str) -> str:
        """
        Fetches the answer from the HTTP JSON response based on the path.

        Args:
            data (dict): HTTP response data.
            key (str): The key path to fetch the value.

        Returns:
            str: The fetched value.
        """
        pattern = re.compile(r'([a-zA-Z_]+)|\[(\d+)\]')
        keys = pattern.findall(key)
        for key_part, index_part in keys:
            if key_part:
                data = data.get(key_part, None)
            elif index_part and isinstance(data, list):
                data = data[int(index_part)] if len(data) > int(index_part) else None
            if data is None:
                return ""
        return data

    def _validate_request(self, *, prompt_request: PromptRequestResponse) -> None:
        """
        Validates the prompt request.

        Args:
            prompt_request (PromptRequestResponse): The prompt request to validate.

        Raises:
            ValueError: If the request is invalid.
        """
        request_pieces: list[PromptRequestPiece] = prompt_request.request_pieces

        if len(request_pieces) != 1:
            raise ValueError("This target only supports a single prompt request piece.")
        if request_pieces[0].original_value_data_type != "text":
            raise ValueError(
                f"This target only supports text prompt input. Got: {type(request_pieces[0].original_value_data_type)}"
            )

    def replace_key(self, data: dict, key: str, replacement: str) -> dict:
        """
        Replaces the value in the HTTP JSON response based on the path.

        Args:
            data (dict): HTTP response data.
            key (str): The key path to replace the value.
            replacement (str): The replacement value.

        Returns:
            dict: The modified data.
        """
        pattern = re.compile(r'([a-zA-Z_]+)|\[(\d+)\]')
        keys = pattern.findall(key)
        result = data

        for index, (key_part, index_part) in enumerate(keys):
            if index != len(keys) - 1:
                if key_part:
                    data = data.get(key_part)
                elif index_part and isinstance(data, list):
                    data = data[int(index_part)] if len(data) > int(index_part) else None
                if data is None:
                    return ""
            else:
                if key_part:
                    data[key_part] = replacement
                else:
                    data[int(index_part)] = replacement

        return result