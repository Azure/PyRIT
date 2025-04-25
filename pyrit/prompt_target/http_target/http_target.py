# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


import json
import logging
import re
from typing import Any, Callable, Optional, Sequence

import httpx

from pyrit.models import (
    PromptRequestPiece,
    PromptRequestResponse,
    construct_response_from_request,
)
from pyrit.prompt_target import PromptTarget, limit_requests_per_minute

logger = logging.getLogger(__name__)


RequestBody = dict[str, Any] | str


class HTTPTarget(PromptTarget):
    """
    HTTP_Target is for endpoints that do not have an API and instead require HTTP request(s) to send a prompt

    Parameters:
        http_request (str): the header parameters as a request (i.e., from Burp)
        prompt_regex_string (str): the placeholder for the prompt
            (default is {PROMPT}) which will be replaced by the actual prompt.
            make sure to modify the http request to have this included, otherwise it will not be properly replaced!
        use_tls: (bool): whether to use TLS or not. Default is True
        callback_function (function): function to parse HTTP response.
            These are the customizable functions which determine how to parse the output
        httpx_client_kwargs: (dict): additional keyword arguments to pass to the HTTP client
    """

    def __init__(
        self,
        http_request: str,
        prompt_regex_string: str = "{PROMPT}",
        use_tls: bool = True,
        callback_function: Callable | None = None,
        max_requests_per_minute: Optional[int] = None,
        **httpx_client_kwargs: Any,
    ) -> None:
        super().__init__(max_requests_per_minute=max_requests_per_minute)
        self.http_request = http_request
        self.callback_function = callback_function
        self.prompt_regex_string = prompt_regex_string
        self.use_tls = use_tls
        self.httpx_client_kwargs = httpx_client_kwargs or {}

    @limit_requests_per_minute
    async def send_prompt_async(self, *, prompt_request: PromptRequestResponse) -> PromptRequestResponse:
        """
        Sends prompt to HTTP endpoint and returns the response
        """

        self._validate_request(prompt_request=prompt_request)
        request = prompt_request.request_pieces[0]

        # Add Prompt into URL (if the URL takes it)
        re_pattern = re.compile(self.prompt_regex_string)
        if re.search(self.prompt_regex_string, self.http_request):
            http_request_w_prompt = re_pattern.sub(request.converted_value, self.http_request)
        else:
            http_request_w_prompt = self.http_request

        header_dict, http_body, url, http_method, http_version = self.parse_raw_http_request(http_request_w_prompt)

        # Make the actual HTTP request:

        # Fix Content-Length if it is in the headers after the prompt is added in:
        if "Content-Length" in header_dict:
            header_dict["Content-Length"] = str(len(http_body))

        http2_version = False
        if http_version and "HTTP/2" in http_version:
            http2_version = True

        async with httpx.AsyncClient(http2=http2_version, **self.httpx_client_kwargs) as client:
            match http_body:
                case dict():
                    response = await client.request(
                        method=http_method,
                        url=url,
                        headers=header_dict,
                        data=http_body,
                        follow_redirects=True,
                    )
                case str():
                    response = await client.request(
                        method=http_method,
                        url=url,
                        headers=header_dict,
                        content=http_body,
                        follow_redirects=True,
                    )
        response_content = response.content

        if self.callback_function:
            response_content = self.callback_function(response=response)

        response_entry = construct_response_from_request(request=request, response_text_pieces=[str(response_content)])

        return response_entry

    def parse_raw_http_request(self, http_request: str) -> tuple[dict[str, str], RequestBody, str, str, str]:
        """
        Parses the HTTP request string into a dictionary of headers

        Parameters:
            http_request: the header parameters as a request str with
                          prompt already injected

        Returns:
            headers_dict (dict): dictionary of all http header values
            body (str): string with body data
            url (str): string with URL
            http_method (str): method (ie GET vs POST)
            http_version (str): HTTP version to use
        """

        headers_dict = {}
        if not http_request:
            return {}, "", "", "", ""

        body = ""

        # Split the request into headers and body by finding the double newlines (\n\n)
        request_parts = http_request.strip().split("\n\n", 1)

        # Parse out the header components
        header_lines = request_parts[0].strip().split("\n")
        http_req_info_line = header_lines[0].split(" ")  # get 1st line like POST /url_ending HTTP_VSN
        header_lines = header_lines[1:]  # rest of the raw request is the headers info

        # Loop through each line and split into key-value pairs
        for line in header_lines:
            key, value = line.split(":", 1)
            headers_dict[key.strip().lower()] = value.strip()

        if "content-length" in headers_dict:
            del headers_dict["content-length"]

        if len(request_parts) > 1:
            # Parse as JSON object if it can be parsed that way
            try:
                body = json.loads(request_parts[1], strict=False)  # Check if valid json
                body = json.dumps(body)
            except json.JSONDecodeError:
                body = request_parts[1]

        if len(http_req_info_line) != 3:
            raise ValueError("Invalid HTTP request line")

        # Capture info from 1st line of raw request
        http_method = http_req_info_line[0]

        url_path = http_req_info_line[1]
        full_url = self._infer_full_url_from_host(path=url_path, headers_dict=headers_dict)

        http_version = http_req_info_line[2]

        return headers_dict, body, full_url, http_method, http_version

    def _infer_full_url_from_host(
        self,
        path: str,
        headers_dict: dict[str, str],
    ) -> str:
        # If path is already a full URL, return it as is
        path = path.lower()
        if path.startswith(("http://", "https://")):
            return path

        http_protocol = "http://"
        if self.use_tls is True:
            http_protocol = "https://"

        host = headers_dict["host"]
        return f"{http_protocol}{host}{path}"

    def _validate_request(self, *, prompt_request: PromptRequestResponse) -> None:
        request_pieces: Sequence[PromptRequestPiece] = prompt_request.request_pieces

        if len(request_pieces) != 1:
            raise ValueError("This target only supports a single prompt request piece.")
