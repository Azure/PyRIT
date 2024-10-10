# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import re


def remove_start_md_json(response_msg: str) -> str:
    """
    Checks the message for the listed start patterns and removes them if present.

    Args:
        response_msg (str): The response message to check.

    Returns:
        str: The response message without the start marker (if one was present).
    """

    start_pattern = re.compile(r"^(```json\n|`json\n|```\n|`\n|```json|`json|```|`|json|json\n)")
    match = start_pattern.match(response_msg)
    if match:
        response_msg = response_msg[match.end() :]

    return response_msg


def remove_end_md_json(response_msg: str) -> str:
    """
    Checks the message for the listed end patterns and removes them if present.

    Args:
        response_msg (str): The response message to check.

    Returns:
        str: The response message without the end marker (if one was present).
    """

    end_pattern = re.compile(r"(\n```|\n`|```|`)$")
    match = end_pattern.search(response_msg)
    if match:
        response_msg = response_msg[: match.start()]

    return response_msg


def extract_json_from_string(response_msg: str) -> str:
    """
    Attempts to extract JSON (object or array) from within a larger string, not specific to markdown.

    Args:
        response_msg (str): The response message to check.

    Returns:
        str: The extracted JSON string if found, otherwise the original string.
    """
    json_pattern = re.compile(r"\{.*\}|\[.*\]")
    match = json_pattern.search(response_msg)
    if match:
        return match.group(0)

    return response_msg
