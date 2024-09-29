# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
import json
from typing import Callable, Union
import requests
from pyrit.prompt_target import PromptTarget
from pyrit.memory import MemoryInterface
from pyrit.models import construct_response_from_request, PromptRequestPiece, PromptRequestResponse
import urllib.parse
import re

logger = logging.getLogger(__name__)


class HTTPTarget(PromptTarget):
    """
    HTTP_Target is for endpoints that do not have an API and instead require HTTP request(s) to send a prompt
    Parameters:
        http_request (str): the header parameters as a request (ie from Burp)
        parse_function (function): function to parse HTTP response
        memory : memory interface
        url_encoding (str): if the prompt is included in the URL, this flag sets how to encode the prompt (ie URL encoding). Defaults to none
        prompt_regex_string (str): the placeholder for the prompt (ie {PROMPT})
        body_encoding (str): encoding for prompt in body (ie url for url encoding)
    """

    def __init__(
        self,
        http_request: str = None,
        parse_function: Callable = None, 
        memory: Union[MemoryInterface, None] = None,
        body_encoding: str = "",
        prompt_regex_string: str = "{PROMPT}"
    ) -> None:

        super().__init__(memory=memory)
        self.http_request = http_request
        self.parse_function = parse_function
        self.body_encoding = body_encoding #TODO: get rid of these
        self.prompt_regex_string = prompt_regex_string

    async def send_prompt_async(self, *, prompt_request: PromptRequestResponse) -> PromptRequestResponse:
        """
        Sends prompt to HTTP endpoint and returns the response
        """

        self._validate_request(prompt_request=prompt_request)
        request = prompt_request.request_pieces[0]
        
        header_dict, http_body, url, http_method = self.parse_raw_http_request()
        re_pattern = re.compile(self.prompt_regex_string)

        #Make the actual HTTP request:

        # Add Prompt into URL (if the URL takes it)
        if self.prompt_regex_string in url:
            prompt_url_safe = urllib.parse.quote(request.original_value) #does url encoding with %20 for spaces for URLs
            formatted_url = re_pattern.sub(prompt_url_safe, self.url)
            self.url = formatted_url

        # Add Prompt into request body (if the body takes it)
        if self.prompt_regex_string in http_body:
            if self.body_encoding == "url":
                encoded_prompt = urllib.parse.urlencode(request.original_value) #does url encoding for query parameters using + for space
                formatted_http_body = re_pattern.sub(encoded_prompt, http_body)

            else:
                formatted_http_body = re_pattern.sub(request.original_value, http_body)
            
            http_body = formatted_http_body
        
        response = requests.request(
            url=url,
            headers=header_dict,
            data=http_body, 
            method=http_method,
            allow_redirects=True # using Requests so we can leave this flag on, rather than httpx
        )

        #parsed_response = self.parse_json_http_response(response)
        parsed_response = self.parse_function(response)

        response_entry = construct_response_from_request(request=request, response_text_pieces=[str(parsed_response)], response_type="text")
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
        http_req_info_line = header_lines[0].split(" ") # get 1st line like POST /url_ending HTTP_VSN
        header_lines = header_lines[1:] # rest of the raw request is the headers info

        # Loop through each line and split into key-value pairs
        for line in header_lines:
            key, value = line.split(":", 1)
            headers_dict[key.strip()] = value.strip()

        if len(request_parts) > 1:
            # Parse as JSON object if it can be parsed that way
            try: 
                body = json.loads(request_parts[1])  # Check if valid json
                body = json.dumps(body)
            except: # if not JSON keep as string
                body = request_parts[1]
            if "Content-Legnth" in headers_dict: 
                headers_dict["Content-Length"] = str(len(body))
        
        # Capture info from 1st line of raw request
        http_method = http_req_info_line[0]
        
        http_url_beg = ""
        if len(http_req_info_line) > 2: 
            http_version = http_req_info_line[2]
            # TODO: qn use_tls_flag variable instead?
            if 'HTTP/2' in http_version:
                http_url_beg = "https://"
            elif 'HTTP/1.1' in http_version:
                http_url_beg = 'http://'
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
        if request_pieces[0].original_value_data_type != "text": #TODO: should this be text or http_request?
            raise ValueError(
                f"This target only supports text prompt input. Got: {type(request_pieces[0].original_value_data_type)}"
            )

def parse_json_http_response(response):
    json_response = json.loads(response.content)
    data_key = fetch_key(data=json_response, key="choices[0].message.content")
    return data_key


def fetch_key(data:dict, key: str) -> str:
    """
    Credit to @Mayuraggarwal1992
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
