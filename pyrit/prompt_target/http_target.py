# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
from typing import Any, Union
import requests
from pyrit.prompt_target import PromptTarget
from pyrit.memory import MemoryInterface
from pyrit.models import construct_response_from_request, PromptRequestPiece, PromptRequestResponse
import urllib.parse
import chardet

logger = logging.getLogger(__name__)


class HTTP_Target(PromptTarget):
    """
    HTTP_Target is for endpoints that do not have an API and instead require HTTP request(s) to send a prompt
    Parameters:
        url (str): URL to send request to
        http_request (str): the header parameters as a request (ie from Burp)
        parse_function (function): function to parse HTTP response
        body (str): HTTP request body
        method (str): HTTP method (eg POST or GET)
        memory : memory interface
        url_encoding (str): if the prompt is included in the URL, this flag sets how to encode the prompt (ie URL encoding). Defaults to none
    """

    def __init__(
        self,
        url: str,
        http_request: str = None,
        parse_function: callable = None, #TODO: this would be where the parse function will go
        body: str = None,
        method: str = "POST",
        memory: Union[MemoryInterface, None] = None,
        url_encoding: str = None,
        body_encoding: str = None
    ) -> None:

        super().__init__(memory=memory)
        self.url = url
        self.http_request = http_request
        self.parse_function = parse_function
        self.body = body
        self.method = method
        self.url_encoding = url_encoding, 
        self.body_encoding = body_encoding

    async def send_prompt_async(self, *, prompt_request: PromptRequestResponse) -> PromptRequestResponse:
        """
        Sends prompt to HTTP endpoint and returns the response
        """

        self._validate_request(prompt_request=prompt_request)
        request = prompt_request.request_pieces[0]
        print("HERE: request: ", request)

        
        request_dict = self.parse_http_request(prompt=str(request.original_value))

        #Make the actual HTTP request:

        # Add Prompt into URL (if the URL takes it)
        if "{PROMPT}" in self.url:
            if self.url_encoding == "url":
                prompt_url_safe = urllib.parse.quote(request.original_value)
                self.url = self.url.replace("{PROMPT}", prompt_url_safe)
            else: 
                self.url = self.url.replace("{PROMPT}", request.original_value)

        # Add Prompt into request body (if the body takes it)
        if "{PROMPT}" in self.body:
            if self.url_encoding:
                encoded_prompt = request.original_value.replace(" ", "+")
                self.body.replace("{PROMPT}", encoded_prompt)
        #TODO: figure out error when using net_utility (says CONTENT LENGTH is not right)
        if self.method == "GET":
            if not request_dict:
                if not self.body:
                    response = requests.get(
                        url=self.url,
                        allow_redirects=True
                )
            else: 
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
        print(type(response.content))
        resp = response.content.decode("utf-8")

        print(response.content)
        response_entry = construct_response_from_request(request=request, response_text_pieces=[str(response.content)], response_type="text")
        return response_entry


    def parse_http_request(self, prompt: str) -> dict[str, str]:
        """
        Parses the HTTP request string into a dictionary of headers
        Parameters:
            prompt (str): prompt to be sent in the HTTP request
        """

        headers_dict = {}
        if not self.http_request:
            return {}
        header_lines = self.http_request.strip().split("\n")
        
        # Loop through each line and split into key-value pairs
        for line in header_lines:
            key, value = line.split(":", 1)
            headers_dict[key.strip()] = value.strip()
        
        headers_dict["Content-Length"] = str(len(self.body))
        return headers_dict
        
        

    def _validate_request(self, *, prompt_request: PromptRequestResponse) -> None:
        request_pieces: list[PromptRequestPiece] = prompt_request.request_pieces

        if len(request_pieces) != 1:
            raise ValueError("This target only supports a single prompt request piece.")
        if request_pieces[0].original_value_data_type != "text": #TODO: should this be text or http_request?
            raise ValueError(
                f"This target only supports text prompt input. Got: {type(request_pieces[0].original_value_data_type)}"
            )