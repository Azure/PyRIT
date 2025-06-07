# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import json
import logging
import os
from abc import ABC
from typing import Callable, Optional

from openai import RateLimitError
from tenacity import (
    retry,
    retry_if_exception_type,
    retry_if_result,
    stop_after_attempt,
    wait_random_exponential,
)

from pyrit.exceptions.exceptions_helpers import log_exception
from pyrit.models.prompt_request_piece import PromptRequestPiece
from pyrit.models.prompt_request_response import (
    PromptRequestResponse,
    construct_response_from_request,
)

# Used with pyrit_custom_result_retry, as this function may be used in conjunction with other decorators
CUSTOM_RESULT_RETRY_MAX_NUM_ATTEMPTS = int(os.getenv("CUSTOM_RESULT_RETRY_MAX_NUM_ATTEMPTS", 10))
RETRY_MAX_NUM_ATTEMPTS = int(os.getenv("RETRY_MAX_NUM_ATTEMPTS", 10))
RETRY_WAIT_MIN_SECONDS = int(os.getenv("RETRY_WAIT_MIN_SECONDS", 5))
RETRY_WAIT_MAX_SECONDS = int(os.getenv("RETRY_WAIT_MAX_SECONDS", 220))

logger = logging.getLogger(__name__)


class PyritException(Exception, ABC):

    def __init__(self, status_code=500, *, message: str = "An error occurred"):
        self.status_code = status_code
        self.message = message
        super().__init__(f"Status Code: {status_code}, Message: {message}")

    def process_exception(self) -> str:
        """
        Logs and returns a string representation of the exception.
        """
        log_message = f"{self.__class__.__name__} encountered: Status Code: {self.status_code}, Message: {self.message}"
        logger.error(log_message)
        # Return a string representation of the exception so users can extract and parse
        return json.dumps({"status_code": self.status_code, "message": self.message})


class BadRequestException(PyritException):
    """Exception class for bad client requests."""

    def __init__(self, status_code: int = 400, *, message: str = "Bad Request"):
        super().__init__(status_code, message=message)


class RateLimitException(PyritException):
    """Exception class for authentication errors."""

    def __init__(self, status_code: int = 429, *, message: str = "Rate Limit Exception"):
        super().__init__(status_code, message=message)


class ServerErrorException(PyritException):
    """Exception class for opaque 5xx errors returned by the server."""

    def __init__(self, status_code: int = 500, *, message: str = "Server Error", body: Optional[str] = None):
        super().__init__(status_code, message=message)
        self.body = body


class EmptyResponseException(BadRequestException):
    """Exception class for empty response errors."""

    def __init__(self, status_code: int = 204, *, message: str = "No Content"):
        super().__init__(status_code=status_code, message=message)


class InvalidJsonException(PyritException):
    """Exception class for blocked content errors."""

    def __init__(self, *, message: str = "Invalid JSON Response"):
        super().__init__(message=message)


class MissingPromptPlaceholderException(PyritException):
    """Exception class for missing prompt placeholder errors."""

    def __init__(self, *, message: str = "No prompt placeholder"):
        super().__init__(message=message)


def pyrit_custom_result_retry(
    retry_function: Callable, retry_max_num_attempts: int = CUSTOM_RESULT_RETRY_MAX_NUM_ATTEMPTS
) -> Callable:
    """
    A decorator to apply retry logic with exponential backoff to a function.

    Retries the function if the result of the retry_function is True,
    with a wait time between retries that follows an exponential backoff strategy.
    Logs retry attempts at the INFO level and stops after a maximum number of attempts.

    Args:
        retry_function (Callable): The boolean function to determine if a retry should occur based
            on the result of the decorated function.
        retry_max_num_attempts (Optional, int): The maximum number of retry attempts. Defaults to
            CUSTOM_RESULT_RETRY_MAX_NUM_ATTEMPTS.
        func (Callable): The function to be decorated.

    Returns:
        Callable: The decorated function with retry logic applied.
    """

    def inner_retry(func):
        global RETRY_WAIT_MIN_SECONDS, RETRY_WAIT_MAX_SECONDS

        return retry(
            reraise=True,
            retry=retry_if_result(retry_function),
            wait=wait_random_exponential(min=RETRY_WAIT_MIN_SECONDS, max=RETRY_WAIT_MAX_SECONDS),
            after=log_exception,
            stop=stop_after_attempt(retry_max_num_attempts),
        )(func)

    return inner_retry


def pyrit_target_retry(func: Callable) -> Callable:
    """
    A decorator to apply retry logic with exponential backoff to a function.

    Retries the function if it raises RateLimitError or EmptyResponseException,
    with a wait time between retries that follows an exponential backoff strategy.
    Logs retry attempts at the INFO level and stops after a maximum number of attempts.

    Args:
        func (Callable): The function to be decorated.

    Returns:
        Callable: The decorated function with retry logic applied.
    """
    global RETRY_MAX_NUM_ATTEMPTS, RETRY_WAIT_MIN_SECONDS, RETRY_WAIT_MAX_SECONDS

    return retry(
        reraise=True,
        retry=retry_if_exception_type(RateLimitError)
        | retry_if_exception_type(EmptyResponseException)
        | retry_if_exception_type(RateLimitException),
        wait=wait_random_exponential(min=RETRY_WAIT_MIN_SECONDS, max=RETRY_WAIT_MAX_SECONDS),
        after=log_exception,
        stop=stop_after_attempt(RETRY_MAX_NUM_ATTEMPTS),
    )(func)


def pyrit_json_retry(func: Callable) -> Callable:
    """
    A decorator to apply retry logic with exponential backoff to a function.

    Retries the function if it raises a JSON error,
    with a wait time between retries that follows an exponential backoff strategy.
    Logs retry attempts at the INFO level and stops after a maximum number of attempts.

    Args:
        func (Callable): The function to be decorated.

    Returns:
        Callable: The decorated function with retry logic applied.
    """
    global RETRY_MAX_NUM_ATTEMPTS, RETRY_WAIT_MIN_SECONDS, RETRY_WAIT_MAX_SECONDS

    return retry(
        reraise=True,
        retry=retry_if_exception_type(InvalidJsonException),
        wait=wait_random_exponential(min=RETRY_WAIT_MIN_SECONDS, max=RETRY_WAIT_MAX_SECONDS),
        after=log_exception,
        stop=stop_after_attempt(RETRY_MAX_NUM_ATTEMPTS),
    )(func)


def pyrit_placeholder_retry(func: Callable) -> Callable:
    """
    A decorator to apply retry logic.

    Retries the function if it raises MissingPromptPlaceholderException.
    Logs retry attempts at the INFO level and stops after a maximum number of attempts.

    Args:
        func (Callable): The function to be decorated.

    Returns:
        Callable: The decorated function with retry logic applied.
    """

    global RETRY_MAX_NUM_ATTEMPTS

    return retry(
        reraise=True,
        retry=retry_if_exception_type(MissingPromptPlaceholderException),
        after=log_exception,
        stop=stop_after_attempt(RETRY_MAX_NUM_ATTEMPTS),
    )(func)


def handle_bad_request_exception(
    response_text: str,
    request: PromptRequestPiece,
    is_content_filter=False,
    error_code: int = 400,
) -> PromptRequestResponse:

    if (
        "content_filter" in response_text
        or "Invalid prompt: your prompt was flagged as potentially violating our usage policy." in response_text
        or is_content_filter
    ):
        # Handle bad request error when content filter system detects harmful content
        bad_request_exception = BadRequestException(error_code, message=response_text)
        resp_text = bad_request_exception.process_exception()
        response_entry = construct_response_from_request(
            request=request, response_text_pieces=[resp_text], response_type="error", error="blocked"
        )
    else:
        raise

    return response_entry
