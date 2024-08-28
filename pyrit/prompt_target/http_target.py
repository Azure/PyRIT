# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
from typing import Any, Union
import requests
from pyrit.prompt_target import PromptTarget
from pyrit.memory import MemoryInterface
from pyrit.models import construct_response_from_request, PromptRequestPiece, PromptRequestResponse

logger = logging.getLogger(__name__)


class HTTP_Target(PromptTarget):
    """
    HTTP_Target is for endpoints that do not have an API and instead require HTTP request(s) to send a prompt
    """

    def __init__(
        self,
        url: str,
        http_request: str = None,
        parse_function: callable = None,
        body: str = None,
        method: str = "POST",
        memory: Union[MemoryInterface, None] = None,
    ) -> None:

        super().__init__(memory=memory)
        self.url = url
        self.http_request = http_request
        self.parse_function = parse_function
        self.body = body
        self.method = method

    async def send_prompt_async(self, *, prompt_request: PromptRequestResponse) -> PromptRequestResponse:
        """
        Sends prompt to HTTP endpoint and returns the response
        """

        self._validate_request(prompt_request=prompt_request)
        request = prompt_request.request_pieces[0]

        request_dict = self.parse_http_request(prompt=str(request.original_value))

        #Make the actual HTTP request:
        if self.method == "GET":
            response = requests.get(
                url=self.url,
                headers=request_dict,
                data=self.body, 
                allow_redirects=True
            )

        elif self.method == "POST":
            if self.body:
                response = requests.post(
                    url=self.url,
                    headers=request_dict,
                    data=self.body,
                    allow_redirects=True
                )
            else:
                response = requests.post(
                    url=self.url,
                    headers=request_dict,
                    allow_redirects=True
                    )
        
        response_entry = construct_response_from_request(request=request, response_text_pieces=[str(response.content)], response_type="text")
        return response_entry


    def parse_http_request(self, prompt: str) -> dict[str, str]:
        """
        Parses the HTTP request string into a dictionary of headers
        Parameters:
            prompt (str): prompt to be sent in the HTTP request
        """

        headers_dict = {}
        header_lines = self.http_request.strip().split("\n")
        
        # Loop through each line and split into key-value pairs
        for line in header_lines:
            key, value = line.split(":", 1)
            headers_dict[key.strip()] = value.strip()
        
        headers_dict["Content-Length"] = str(len(prompt))
        return headers_dict
        
        

    def _validate_request(self, *, prompt_request: PromptRequestResponse) -> None:
        request_pieces: list[PromptRequestPiece] = prompt_request.request_pieces

        if len(request_pieces) != 1:
            raise ValueError("This target only supports a single prompt request piece.")
        if request_pieces[0].original_value_data_type != "text": #TODO: should this be text or http_request?
            raise ValueError(
                f"This target only supports text prompt input. Got: {type(request_pieces[0].original_value_data_type)}"
            )