# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
from typing import Any, Union
import requests
from pyrit.prompt_target import PromptTarget
from pyrit.memory import MemoryInterface
from pyrit.models import construct_response_from_request, PromptRequestPiece, PromptRequestResponse
import urllib.parse

logger = logging.getLogger(__name__)


class HTTPTarget(PromptTarget):
    """
    HTTP_Target is for endpoints that do not have an API and instead require HTTP request(s) to send a prompt
    Parameters:
        http_request (str): the header parameters as a request (ie from Burp)
        parse_function (function): function to parse HTTP response
        memory : memory interface
        url_encoding (str): if the prompt is included in the URL, this flag sets how to encode the prompt (ie URL encoding). Defaults to none
    """

    def __init__(
        self,
        http_request: str = None,
        parse_function: callable = None, #TODO: this would be where the parse function will go
        memory: Union[MemoryInterface, None] = None,
        url_encoding: str = None,
        body_encoding: str = None
    ) -> None:

        super().__init__(memory=memory)
        self.http_request = http_request
        self.parse_function = parse_function
        self.url_encoding = url_encoding, 
        self.body_encoding = body_encoding #TODO: get rid of these

    async def send_prompt_async(self, *, prompt_request: PromptRequestResponse) -> PromptRequestResponse:
        """
        Sends prompt to HTTP endpoint and returns the response
        """

        self._validate_request(prompt_request=prompt_request)
        request = prompt_request.request_pieces[0]
        
        header_dict, http_body, url, http_method = self.parse_http_request()

        #Make the actual HTTP request:

        # Add Prompt into URL (if the URL takes it)
        if "{PROMPT}" in url:
            if self.url_encoding == "url": #TODO: get rid of & move to converters
                prompt_url_safe = urllib.parse.quote(request.original_value)
                self.url = url.replace("{PROMPT}", prompt_url_safe)
            else: 
                self.url = url.replace("{PROMPT}", request.original_value)

        # Add Prompt into request body (if the body takes it)
        if "{PROMPT}" in http_body:
            if self.url_encoding:
                encoded_prompt = request.original_value.replace(" ", "+")
                http_body.replace("{PROMPT}", encoded_prompt)
        
        #TODO: include vsn here
        response = requests.request(
            url=url,
            headers=header_dict,
            data=http_body, 
            method=http_method,
            allow_redirects=True # using Requests so we can leave this flag on, rather than httpx
        )

        response_entry = construct_response_from_request(request=request, response_text_pieces=[str(response.content)], response_type="text")
        return response_entry


    def parse_http_request(self):
        """
        Parses the HTTP request string into a dictionary of headers
        Returns:
            headers_dict (dict): dictionary of all http header values
            body (str): string with body data
            url (str): string with URL
            http_method (str): method (ie GET vs POST)
        """

        headers_dict = {}
        if not self.http_request:
            return {}, "", "", ""
        
        body = ""

        # Split the request into headers and body by finding the double newlines (\n\n)
        request_parts = self.http_request.strip().split("\n\n", 1)

        # Parse out the header components
        header_lines = request_parts[0].strip().split("\n")
        http_req_info_line = header_lines[0].split(" ") # get 1st line like POST /url_ending HTTP_VSN
        header_lines = header_lines[1:] # rest of the raw request is the headers info

        # Loop through each line and split into key-value pairs
        for line in header_lines:
            key, value = line.split(":", 1)
            headers_dict[key.strip()] = value.strip()

        if len(request_parts) > 1:
            body = request_parts[1]
            headers_dict["Content-Length"] = str(len(body))
        
        # Capture info from 1st line of raw request
        http_method = http_req_info_line[0]
        http_version = http_req_info_line[1]
        http_url_beg = ""
        if 'HTTP/2' in http_version:
            http_url_beg = "https://"
        elif 'HTTP/1.1' in http_version:
            http_url_beg = 'http://'
        else:
                raise ValueError(f"Unsupported protocol: {http_version}")

        url = http_url_beg + headers_dict["Host"] + http_req_info_line[1]

        return headers_dict, body, url, http_method
        
        
    def _validate_request(self, *, prompt_request: PromptRequestResponse) -> None:
        request_pieces: list[PromptRequestPiece] = prompt_request.request_pieces

        if len(request_pieces) != 1:
            raise ValueError("This target only supports a single prompt request piece.")
        if request_pieces[0].original_value_data_type != "text": #TODO: should this be text or http_request?
            raise ValueError(
                f"This target only supports text prompt input. Got: {type(request_pieces[0].original_value_data_type)}"
            )