# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


import json
import re
from typing import Callable

import requests


def get_http_target_json_response_callback_function(key: str) -> Callable:
    """
    Purpose: determines proper parsing response function for an HTTP Request
    Parameters:
        key (str): this is the path pattern to follow for parsing the output response
            (ie for AOAI this would be choices[0].message.content)
            (for BIC this needs to be a regex pattern for the desired output)
        response_type (ResponseType): this is the type of response (ie HTML or JSON)
    Returns: proper output parsing response
    """

    def parse_json_http_response(response: requests.Response):
        """
        Purpose: parses json outputs
        Parameters:
            response (response): the HTTP Response to parse
        Returns: parsed output from response given a "key" path to follow
        """
        json_response = json.loads(response.content)
        data_key = _fetch_key(data=json_response, key=key)
        return data_key

    return parse_json_http_response


def get_http_target_regex_matching_callback_function(key: str, url: str = None) -> Callable:
    def parse_using_regex(response: requests.Response):
        """
        Purpose: parses text outputs using regex
        Parameters:
            url (optional str): the original URL if this is needed to get a full URL response back (ie BIC)
            key (str): this is the regex pattern to follow for parsing the output response
            response (response): the HTTP Response to parse
        Returns: parsed output from response given a regex pattern to follow
        """
        re_pattern = re.compile(key)
        match = re.search(re_pattern, str(response.content))
        if match:
            if url:
                return url + match.group()
            else:
                return match.group()
        else:
            return str(response.content)

    return parse_using_regex


def _fetch_key(data: dict, key: str):
    """
    Credit to @Mayuraggarwal1992
    Fetches the answer from the HTTP JSON response based on the path.

    Args:
        data (dict): HTTP response data.
        key (str): The key path to fetch the value.

    Returns:
        str: The fetched value.
    """
    pattern = re.compile(r"([a-zA-Z_]+)|\[(\d+)\]")
    keys = pattern.findall(key)
    for key_part, index_part in keys:
        if key_part:
            data = data.get(key_part, None)
        elif index_part and isinstance(data, list):
            data = data[int(index_part)] if len(data) > int(index_part) else None
        if data is None:
            return ""
    return data
