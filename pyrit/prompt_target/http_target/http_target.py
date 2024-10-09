# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import json
import logging
import re
import requests
from typing import Callable, Union

from pyrit.memory import MemoryInterface
from pyrit.models import construct_response_from_request, PromptRequestPiece, PromptRequestResponse
from pyrit.prompt_target import PromptTarget

logger = logging.getLogger(__name__)


class HTTPTarget(PromptTarget):
    """
    HTTP_Target is for endpoints that do not have an API and instead require HTTP request(s) to send a prompt
    Parameters:
        http_request (str): the header parameters as a request (ie from Burp)
        prompt_regex_string (str): the placeholder for the prompt
            (default is {PROMPT}) which will be replaced by the actual prompt.
            make sure to modify the http request to have this included, otherwise it will not be properly replaced!
        use_tls: (bool): whether to use TLS or not. Default is True
        callback_function (function): function to parse HTTP response.
            These are the customizable functions which determine how to parse the output
        memory : memory interface
    """

    def __init__(
        self,
        http_request: str = None,
        prompt_regex_string: str = "{PROMPT}",
        use_tls: bool = True,
        callback_function: Callable = None,
        memory: Union[MemoryInterface, None] = None,
    ) -> None:

        super().__init__(memory=memory)
        self.http_request = http_request
        self.callback_function = callback_function
        self.prompt_regex_string = prompt_regex_string
        self.use_tls = use_tls

        if not self.http_request:
            raise ValueError("HTTP Request is required for HTTP Target")

    async def send_prompt_async(self, *, prompt_request: PromptRequestResponse) -> PromptRequestResponse:
        """
        Sends prompt to HTTP endpoint and returns the response
        """

        self._validate_request(prompt_request=prompt_request)
        request = prompt_request.request_pieces[0]

        header_dict, http_body, url, http_method = self.parse_raw_http_request()
        re_pattern = re.compile(self.prompt_regex_string)

        # Make the actual HTTP request:

        # Add Prompt into URL (if the URL takes it)
        if re.search(self.prompt_regex_string, url):
            # by default doing URL encoding for prompts that go in URL
            url = re_pattern.sub(request.converted_value, url)

        # Add Prompt into request body (if the body takes it)
        if re.search(self.prompt_regex_string, http_body):
            http_body = re_pattern.sub(request.converted_value, http_body)  #

        response = requests.request(
            method=http_method,
            url=url,
            headers=header_dict,
            data=http_body,
            allow_redirects=True,  # This is defaulted to true but using requests over httpx for this reason
        )

        if self.callback_function:
            parsed_response = self.callback_function(response=response)
            response_entry = construct_response_from_request(
                request=request, response_text_pieces=[str(parsed_response)]
            )

        else:
            response_entry = construct_response_from_request(
                request=request, response_text_pieces=[str(response.content)]
            )
        return response_entry

    def parse_raw_http_request(self):
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
        http_req_info_line = header_lines[0].split(" ")  # get 1st line like POST /url_ending HTTP_VSN
        header_lines = header_lines[1:]  # rest of the raw request is the headers info

        # Loop through each line and split into key-value pairs
        for line in header_lines:
            key, value = line.split(":", 1)
            headers_dict[key.strip()] = value.strip()

        if len(request_parts) > 1:
            # Parse as JSON object if it can be parsed that way
            try:
                body = json.loads(request_parts[1], strict=False)  # Check if valid json
                body = json.dumps(body)
            except json.JSONDecodeError:
                body = request_parts[1]
            if "Content-Length" in headers_dict:
                headers_dict["Content-Length"] = str(len(body))

        # Capture info from 1st line of raw request
        http_method = http_req_info_line[0]

        http_url_beg = ""
        if len(http_req_info_line) > 2:
            http_version = http_req_info_line[2]
            if "HTTP/2" in http_version or "HTTP/1.1" in http_version:
                if self.use_tls == True:
                    http_url_beg = "https://"
                else:
                    http_url_beg = "http://"
            else:
                raise ValueError(f"Unsupported protocol: {http_version}")

        url = ""
        if http_url_beg and "http" not in http_req_info_line[1]:
            url = http_url_beg
        if "Host" in headers_dict.keys():
            url += headers_dict["Host"]
        url += http_req_info_line[1]

        return headers_dict, body, url, http_method

    def _validate_request(self, *, prompt_request: PromptRequestResponse) -> None:
        request_pieces: list[PromptRequestPiece] = prompt_request.request_pieces

        if len(request_pieces) != 1:
            raise ValueError("This target only supports a single prompt request piece.")
