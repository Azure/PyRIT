# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import json
import re
import time
import logging

from tenacity import RetryCallState

logger = logging.getLogger(__name__)


def log_exception(retry_state: RetryCallState):
    # Log each retry attempt with exception details at ERROR level
    elapsed_time = time.monotonic() - retry_state.start_time
    call_count = retry_state.attempt_number

    if retry_state.outcome.failed:
        exception = retry_state.outcome.exception()
        logger.error(
            f"Retry attempt {call_count} for {retry_state.fn.__name__} failed with exception: {exception}. "
            f"Elapsed time: {elapsed_time} seconds. Total calls: {call_count}"
        )


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


def remove_markdown_json(response_msg: str) -> str:
    """
    Checks if the response message is in JSON format and removes Markdown formatting if present.

    Args:
        response_msg (str): The response message to check.

    Returns:
        str: The response message without Markdown formatting if present.
    """

    response_msg = remove_start_md_json(response_msg)
    response_msg = remove_end_md_json(response_msg)

    # Validate if the remaining response message is valid JSON. If it's still not valid
    # after removing the markdown notation, try to extract JSON from within the string.
    try:
        json.loads(response_msg)
        return response_msg
    except json.JSONDecodeError:
        response_msg = extract_json_from_string(response_msg)
        try:
            json.loads(response_msg)
            return response_msg
        except json.JSONDecodeError:
            return "Invalid JSON response: {}".format(response_msg)
