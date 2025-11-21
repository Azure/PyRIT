# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Shared error handling utilities for OpenAI SDK interactions.

This module provides defensive error parsing, request ID extraction, and retry-after
hint extraction for consistent error handling across OpenAI-based prompt targets.
"""

import json
import logging
from typing import Any, Callable, List, Optional, Tuple, Union

from openai import BadRequestError, ContentFilterFinishReasonError, RateLimitError
from openai._exceptions import APIConnectionError, APIStatusError, APITimeoutError, AuthenticationError

from pyrit.exceptions import EmptyResponseException, PyritException
from pyrit.exceptions.exception_classes import RateLimitException, handle_bad_request_exception
from pyrit.models import Message, MessagePiece, construct_response_from_request

logger = logging.getLogger(__name__)


def _extract_request_id_from_exception(exc: Exception) -> Optional[str]:
    """
    Extract the x-request-id from an OpenAI SDK exception for logging/telemetry.
    
    Args:
        exc: An exception from the OpenAI SDK (e.g., BadRequestError, RateLimitError).
        
    Returns:
        The request ID string if found, otherwise None.
    """
    try:
        resp = getattr(exc, "response", None)
        if resp is not None:
            # Try both common header name variants
            return resp.headers.get("x-request-id") or resp.headers.get("X-Request-Id")
    except Exception:
        pass
    return None


def _extract_retry_after_from_exception(exc: Exception) -> Optional[float]:
    """
    Extract the Retry-After header from a rate-limit exception for intelligent backoff.
    
    Args:
        exc: A rate-limit exception from the OpenAI SDK.
        
    Returns:
        The retry-after value in seconds as a float, or None if not present.
    """
    try:
        resp = getattr(exc, "response", None)
        if resp is not None:
            ra = resp.headers.get("retry-after") or resp.headers.get("Retry-After")
            if ra is not None:
                try:
                    return float(ra)
                except ValueError:
                    # Retry-After can be an HTTP date string; ignore for now
                    return None
    except Exception:
        pass
    return None


def _extract_error_payload(exc: Exception) -> Tuple[Union[dict, str], bool]:
    """
    Extract error payload and detect content filter from an OpenAI SDK exception.
    
    This function tries multiple strategies to parse error information:
    1. Try response.json() if response object exists
    2. Fall back to e.body attribute
    3. Fall back to str(e)
    
    It also attempts to detect whether the error is due to content filtering by:
    - Checking for error.code == "content_filter"
    - Searching for "content_filter" or "policy" keywords in the payload
    
    Args:
        exc: An exception from the OpenAI SDK (typically BadRequestError).
        
    Returns:
        A tuple of (payload, is_content_filter) where:
        - payload is either a dict (if JSON) or a string
        - is_content_filter is True if the error appears to be content policy related
    """
    # Strategy 1: Try response JSON
    resp = getattr(exc, "response", None)
    if resp is not None:
        try:
            data = resp.json()
            # Check for explicit content_filter code
            code = (data.get("error") or {}).get("code")
            is_cf = (code == "content_filter")
            # Heuristic: Azure sometimes uses other codes with policy-related content
            if not is_cf and "content_filter" in json.dumps(data).lower():
                is_cf = True
            return data, is_cf
        except Exception:
            # Try text fallback from response
            try:
                text = resp.text
            except Exception:
                text = None
            if text:
                lower = text.lower()
                is_cf = "content_filter" in lower or "policy" in lower
                return text, is_cf

    # Strategy 2: Try e.body attribute
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        code = (body.get("error") or {}).get("code")
        is_cf = (code == "content_filter")
        if not is_cf and "content_filter" in json.dumps(body).lower():
            is_cf = True
        return body, is_cf
    elif isinstance(body, str):
        try:
            data = json.loads(body)
            code = (data.get("error") or {}).get("code")
            is_cf = (code == "content_filter")
            if not is_cf and "content_filter" in json.dumps(data).lower():
                is_cf = True
            return data, is_cf
        except json.JSONDecodeError:
            lower = body.lower()
            is_cf = "content_filter" in lower or "policy" in lower
            return body, is_cf

    # Strategy 3: Fall back to str(e)
    text = str(exc)
    lower = text.lower()
    is_cf = "content_filter" in lower or "policy" in lower
    return text, is_cf


def _check_chat_completion_content_filter(completion: Any) -> bool:
    """
    Check if a Chat Completions API response has finish_reason=content_filter.
    
    Args:
        completion: A ChatCompletion object from the OpenAI SDK.
        
    Returns:
        True if content was filtered, False otherwise.
    """
    try:
        if completion.choices and completion.choices[0].finish_reason == "content_filter":
            return True
    except (AttributeError, IndexError):
        pass
    return False


def _check_response_api_content_filter(response: Any) -> bool:
    """
    Check if a Response API response has a content filter error.
    
    The Response API indicates content filtering differently from Chat Completions:
    - Status is None or "failed"
    - Error object has code="content_filter"
    
    Args:
        response: A Response object from the OpenAI SDK.
        
    Returns:
        True if content was filtered, False otherwise.
    """
    try:
        # Response API uses error.code for content filtering, not finish_reason
        # Access the typed ResponseError object directly
        if hasattr(response, 'error') and response.error is not None:
            if response.error.code == 'content_filter':
                return True
        
        # Also check if status indicates failure with no error (defensive)
        if hasattr(response, 'status') and response.status in [None, 'failed']:
            if hasattr(response, 'error') and response.error is not None:
                return True
    except (AttributeError, TypeError):
        pass
    return False


def construct_chat_completion_message(
    *,
    completion_response: Any,
    message_piece: MessagePiece,
) -> Any:
    """
    Shared logic to construct a Message from a Chat Completions API response.
    
    Handles common scenarios:
    - Extracting content from response.choices[0].message.content
    - Handling finish_reason (stop, length, content_filter)
    - Empty response validation
    
    Args:
        completion_response: The completion response from OpenAI SDK (ChatCompletion object).
        message_piece: The original request message piece.
        
    Returns:
        Message: The constructed message.
        
    Raises:
        PyritException: For unexpected response structures or finish reasons.
        EmptyResponseException: When the API returns an empty response.
    """
    # Extract the finish reason and content from the SDK response
    if not completion_response.choices:
        raise PyritException(message="No choices returned in the completion response.")
        
    choice = completion_response.choices[0]
    finish_reason = choice.finish_reason
    extracted_response: str = ""
    
    # finish_reason="stop" means API returned complete message
    # "length" means API returned incomplete message due to max_tokens limit
    if finish_reason in ["stop", "length"]:
        extracted_response = choice.message.content or ""

        # Handle empty response
        if not extracted_response:
            logger.error("The chat returned an empty response.")
            raise EmptyResponseException(message="The chat returned an empty response.")
    elif finish_reason == "content_filter":
        # Content filter with status 200 indicates that the model output was filtered
        # Note: The SDK should raise ContentFilterFinishReasonError for this case,
        # but we handle it here as a fallback
        return handle_bad_request_exception(
            response_text=completion_response.model_dump_json(), 
            request=message_piece, 
            error_code=200, 
            is_content_filter=True
        )
    else:
        raise PyritException(
            message=f"Unknown finish_reason {finish_reason} from response: {completion_response.model_dump_json()}"
        )

    return construct_response_from_request(request=message_piece, response_text_pieces=[extracted_response])


def construct_response_message(
    *,
    completion_response: Any,
    message_piece: MessagePiece,
    parse_section_fn: Callable,
) -> Any:
    """
    Shared logic to construct a Message from a Response API response.
    
    Parses the Response API structure with status, error, and output sections
    directly from the strongly-typed SDK response.
    
    Args:
        completion_response: The Response object from the OpenAI SDK.
        message_piece: The original request message piece.
        parse_section_fn: Function to parse individual output sections.
        
    Returns:
        Message: The constructed message.
        
    Raises:
        PyritException: For unexpected response structures.
        EmptyResponseException: When the API returns no valid message pieces.
    """
    # Check for error response - error is a ResponseError object or None
    if completion_response.error is not None:
        # error.code should be "content_filter" for content filtering
        if completion_response.error.code == "content_filter":
            logger.warning("Output content filtered by content policy.")
            return handle_bad_request_exception(
                response_text=completion_response.model_dump_json(),
                request=message_piece,
                error_code=200,
                is_content_filter=True,
            )
        # Other error cases
        raise PyritException(
            message=f"Response error: {completion_response.error.code} - {completion_response.error.message}"
        )

    # Check status - should be "completed" for successful responses
    if completion_response.status != "completed":
        raise PyritException(message=f"Unexpected status: {completion_response.status}")

    # Extract message pieces from the output sections
    extracted_response_pieces: List[MessagePiece] = []
    
    if completion_response.output:
        for section in completion_response.output:
            piece = parse_section_fn(
                section=section,
                message_piece=message_piece,
                error=None,  # error is already handled above
            )
            if piece is None:
                continue
            extracted_response_pieces.append(piece)

    # Check for empty response
    if not extracted_response_pieces:
        logger.error("The response returned no valid message pieces.")
        raise EmptyResponseException(message="The chat returned an empty response.")

    return Message(message_pieces=extracted_response_pieces)


def _parse_openai_json_response(open_ai_str_response: str) -> dict:
    """
    Parse OpenAI JSON response with consistent error handling.
    
    Args:
        open_ai_str_response: JSON string response from OpenAI API.
        
    Returns:
        Parsed JSON response as a dictionary.
        
    Raises:
        PyritException: If JSON parsing fails.
    """
    try:
        return json.loads(open_ai_str_response)
    except json.JSONDecodeError as e:
        response_start = open_ai_str_response[:100]
        raise PyritException(
            message=f"Failed to parse JSON response. Please check your endpoint.\n"
            f"Response: {response_start}\nFull error: {e}"
        )


def _handle_content_filter_response(
    open_ai_str_response: str,
    message_piece: MessagePiece,
) -> Any:
    """
    Unified content filter response handler for both Chat Completions and Response APIs.
    
    Returns an error Message when content filtering is detected.
    
    Args:
        open_ai_str_response: The raw JSON response string.
        message_piece: The original request message piece.
        
    Returns:
        Message with error indication for content filtering.
    """
    logger.warning("Output content filtered by content policy.")
    return handle_bad_request_exception(
        response_text=open_ai_str_response,
        request=message_piece,
        error_code=200,  # Content filter with status 200
        is_content_filter=True,
    )


def construct_message_from_openai_json(
    *,
    open_ai_str_response: str,
    message_piece: MessagePiece,
) -> Any:
    """
    Legacy method for backward compatibility with JSON string responses.
    
    Parses a JSON string response from OpenAI Chat Completions API.
    Modern implementations should use construct_chat_completion_message with SDK response objects.
    
    Args:
        open_ai_str_response: JSON string response from OpenAI API.
        message_piece: The original request message piece.
        
    Returns:
        Message: The constructed message.
        
    Raises:
        PyritException: For JSON parsing errors or unexpected response structures.
        EmptyResponseException: When the API returns an empty response.
    """
    response = _parse_openai_json_response(open_ai_str_response)

    finish_reason = response["choices"][0]["finish_reason"]
    
    # Check for content filter
    if finish_reason == "content_filter":
        return _handle_content_filter_response(open_ai_str_response, message_piece)
    
    # finish_reason="stop" means API returned complete message
    # "length" means API returned incomplete message due to max_tokens limit
    if finish_reason not in ["stop", "length"]:
        raise PyritException(message=f"Unknown finish_reason {finish_reason} from response: {response}")
    
    extracted_response = response["choices"][0]["message"]["content"]

    # Check for empty response
    if not extracted_response:
        logger.error("The chat returned an empty response.")
        raise EmptyResponseException(message="The chat returned an empty response.")

    return construct_response_from_request(request=message_piece, response_text_pieces=[extracted_response])


async def handle_openai_completion_with_errors(
    make_request_fn: Callable,
    message_piece: MessagePiece,
    check_content_filter_fn: Callable[[Any], bool],
    construct_message_fn: Callable,
    post_process_fn: Optional[Callable[[Any], Any]] = None,
    *args,
    **kwargs
) -> Any:
    """
    Unified error handling wrapper for OpenAI SDK completion requests.
    
    This function wraps the OpenAI SDK call and handles all common error scenarios:
    - Content filtering (both proactive checks and SDK exceptions)
    - Bad request errors (400s with content filter detection)
    - Rate limiting (429s with retry-after extraction)
    - API status errors (other HTTP errors)
    - Transient errors (timeouts, connection issues)
    - Authentication errors
    
    The function supports both simple request-response flows (Chat Completions) and 
    complex flows with post-processing (Response API with tool calls).
    
    Args:
        make_request_fn: Async function that makes the OpenAI SDK request and returns the completion.
        message_piece: The MessagePiece being processed (for error responses).
        check_content_filter_fn: Function to check if the completion has content_filter error.
        construct_message_fn: Function to convert completion response to Message format.
        post_process_fn: Optional function to post-process the constructed message (e.g., extract tool calls).
        *args, **kwargs: Additional arguments to pass to make_request_fn.
        
    Returns:
        The constructed Message from the completion (optionally post-processed), or an error Message.
        
    Raises:
        RateLimitException: For 429 rate limit errors.
        Various OpenAI SDK exceptions: For non-recoverable errors.
    """
    try:
        completion = await make_request_fn(*args, **kwargs)

        # Proactively check for content_filter in the response
        # The SDK doesn't always raise ContentFilterFinishReasonError for standard completions
        if check_content_filter_fn(completion):
            logger.warning("Output content filtered (detected in response)")
            return handle_bad_request_exception(
                response_text="Output filtered by content policy.",
                request=message_piece,
                error_code=200,  # Soft-fail with 200 for output filtering
                is_content_filter=True,
            )

        # Convert the SDK response to Message format
        result = construct_message_fn(completion_response=completion, message_piece=message_piece)
        
        # Apply optional post-processing (e.g., extract tool calls for Response API)
        if post_process_fn is not None:
            result = post_process_fn(result)
        
        return result

    except ContentFilterFinishReasonError as e:
        # Content filter error raised by SDK during parse/structured output flows
        request_id = _extract_request_id_from_exception(e)
        logger.error(f"Content filter error (SDK raised). request_id={request_id} error={e}")
        return handle_bad_request_exception(
            response_text=str(e),
            request=message_piece,
            error_code=200,  # Content filter with 200 status
            is_content_filter=True,
        )
    except BadRequestError as e:
        # Handle 400 errors - includes input policy filters and some Azure output-filter 400s
        payload, is_content_filter = _extract_error_payload(e)
        request_id = _extract_request_id_from_exception(e)
        
        # Safely serialize payload for logging
        try:
            payload_str = payload if isinstance(payload, str) else json.dumps(payload)[:200]
        except (TypeError, ValueError):
            # If JSON serialization fails (e.g., contains non-serializable objects), use str()
            payload_str = str(payload)[:200]
        
        logger.warning(
            f"BadRequestError request_id={request_id} is_content_filter={is_content_filter} "
            f"payload={payload_str}"
        )

        return handle_bad_request_exception(
            response_text=str(payload),
            request=message_piece,
            error_code=400,
            is_content_filter=is_content_filter,
        )
    except RateLimitError as e:
        # SDK's RateLimitError (429)
        request_id = _extract_request_id_from_exception(e)
        retry_after = _extract_retry_after_from_exception(e)
        logger.warning(f"RateLimitError request_id={request_id} retry_after={retry_after} error={e}")
        raise RateLimitException()
    except APIStatusError as e:
        # Other API status errors - check for 429 here as well
        request_id = _extract_request_id_from_exception(e)
        if getattr(e, "status_code", None) == 429:
            retry_after = _extract_retry_after_from_exception(e)
            logger.warning(f"429 via APIStatusError request_id={request_id} retry_after={retry_after}")
            raise RateLimitException()
        else:
            logger.exception(
                f"APIStatusError request_id={request_id} status={getattr(e, 'status_code', None)} error={e}"
            )
            raise
    except (APITimeoutError, APIConnectionError) as e:
        # Transient infrastructure errors - these are retryable
        request_id = _extract_request_id_from_exception(e)
        logger.warning(f"Transient API error ({e.__class__.__name__}) request_id={request_id} error={e}")
        raise
    except AuthenticationError as e:
        # Authentication errors - non-retryable, surface quickly
        request_id = _extract_request_id_from_exception(e)
        logger.error(f"Authentication error request_id={request_id} error={e}")
        raise
