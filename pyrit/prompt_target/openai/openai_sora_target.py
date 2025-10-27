# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import json
import logging
import os
from enum import Enum
from typing import Any, Dict, Optional

import httpx

from pyrit.common import net_utility
from pyrit.exceptions import (
    RateLimitException,
    handle_bad_request_exception,
    pyrit_custom_result_retry,
    pyrit_target_retry,
)
from pyrit.models import (
    Message,
    MessagePiece,
    PromptResponseError,
    construct_response_from_request,
    data_serializer_factory,
)
from pyrit.prompt_target import OpenAITarget, limit_requests_per_minute

logger = logging.getLogger(__name__)


class JobStatus(Enum):
    PREPROCESSING = "preprocessing"
    QUEUED = "queued"
    PROCESSING = "processing"
    IN_PROGRESS = "in_progress"  # For Sora-2 API
    SUCCEEDED = "succeeded"
    COMPLETED = "completed"  # For Sora-2 API
    FAILED = "failed"
    CANCELLED = "cancelled"


class FailureReason(Enum):
    INPUT_MODERATION = "input_moderation"
    INTERNAL_ERROR = "internal_error"
    OUTPUT_MODERATION = "output_moderation"


class OpenAISoraTarget(OpenAITarget):
    """
    Unified OpenAI Sora Target supporting both Sora-1 and Sora-2 APIs.

    API version is automatically detected from the endpoint URL:
    - Sora-2: Endpoint contains "v1/videos"
    - Sora-1: All other endpoints

    Provider detection (OpenAI vs Azure OpenAI):
    - Azure OpenAI: Endpoint contains "azure.com"
    - OpenAI: All other endpoints (e.g., "openai.com")

    Sora-1 API:
    - Uses JSON body with /jobs endpoints
    - Supported resolutions: 480x480, 854x480, 720x720, 1280x720, 1080x1080, 1920x1080
    - Duration: up to 20s (10s for 1080p)

    Sora-2 API:
    - Uses multipart form data with direct task endpoints
    - Supported resolutions:

      * Sora-2 (both Azure OpenAI and OpenAI): 720x1280, 1280x720
      * Sora-2-Pro (OpenAI only): 720x1280, 1280x720, 1024x1792, 1792x1024

    - Duration: 4, 8, or 12 seconds only

    Default resolution (1280x720) works with both APIs on OpenAI.
    Default duration (4s) works with both APIs (v1 supports up to 20s, v2 requires 4/8/12s).
    """

    # Maximum number of retries for check_job_status_async()
    CHECK_JOB_RETRY_MAX_NUM_ATTEMPTS: int = int(os.getenv("CUSTOM_RESULT_RETRY_MAX_NUM_ATTEMPTS", "120"))

    # Resolution sets
    V1_RESOLUTIONS = ["480x480", "854x480", "720x720", "1280x720", "1080x1080", "1920x1080"]

    # Sora v2 resolutions:
    # - Sora-2 (both Azure OpenAI and OpenAI): 720x1280, 1280x720
    # - Sora-2-Pro (OpenAI only): 720x1280, 1280x720, 1024x1792, 1792x1024
    V2_RESOLUTIONS = ["720x1280", "1280x720", "1024x1792", "1792x1024"]

    V2_DURATIONS = [4, 8, 12]

    # Utility functions which define when to retry calls to check status and download video
    @staticmethod
    def _should_retry_check_job(response: httpx.Response) -> bool:
        """
        Returns True if the job status is not SUCCEEDED, COMPLETED, FAILED, or CANCELLED.
        """
        content = json.loads(response.content)
        status = content.get("status", None)
        should_retry = status not in [
            JobStatus.SUCCEEDED.value,
            JobStatus.COMPLETED.value,
            JobStatus.FAILED.value,
            JobStatus.CANCELLED.value,
        ]

        # Log detailed information when status check is about to retry or fail
        if should_retry:
            logger.debug(f"Job status '{status}' requires retry. Full response: {content}")

        return should_retry

    @staticmethod
    def _should_retry_video_download(response: httpx.Response) -> bool:
        """
        Returns True if the video download status is not 200 (success).
        """
        return response.status_code != 200

    def __init__(
        self,
        *,
        resolution_dimensions: str = "1280x720",
        n_seconds: int = 4,
        **kwargs,
    ):
        """Initialize the unified OpenAI Sora Target.

        Args:
            model_name (str, Optional): The name of the model.
            endpoint (str, Optional): The target URL for the OpenAI service.
            api_key (str, Optional): The API key for accessing the service.
                Uses OPENAI_SORA_KEY environment variable by default.
            headers (str, Optional): Extra headers of the endpoint (JSON).
            use_entra_auth (bool, Optional): When set to True, user authentication is used
                instead of API Key. DefaultAzureCredential is taken for
                https://cognitiveservices.azure.com/.default. Please run `az login` locally
                to leverage user AuthN.
            max_requests_per_minute (int, Optional): Number of requests the target can handle per
                minute before hitting a rate limit.
            httpx_client_kwargs (dict, Optional): Additional kwargs to be passed to the
                `httpx.AsyncClient()` constructor.
            resolution_dimensions (str, Optional): Resolution dimensions for the video in WIDTHxHEIGHT format.
                Defaults to "1280x720" which works with Sora-1 and Sora-2 on both providers.
                Sora-1 supported resolutions: "480x480", "854x480",
                "720x720", "1280x720", "1080x1080", "1920x1080".
                Sora-2 supported resolutions (both OpenAI and Azure OpenAI per API spec):
                "720x1280", "1280x720"
                Sora-2-Pro (OpenAI only) supported resolutions:
                "720x1280", "1280x720", "1024x1792", "1792x1024"
            n_seconds (int, Optional): The duration of the generated video (in seconds).
                Defaults to 4 (compatible with both APIs).
                Sora-1 supports up to 20 seconds (10 seconds max for 1080p resolutions).
                Sora-2 supports exactly 4, 8, or 12 seconds.
        """
        # Initialize parent class first to get endpoint
        super().__init__(**kwargs)

        # Detect API version
        self._detected_api_version = self._detect_api_version()

        # Set instance variables
        self._n_seconds = n_seconds
        self._validate_duration()
        self._width, self._height = self._parse_and_validate_resolution(resolution_dimensions=resolution_dimensions)
        self._params: Dict[str, Any] = {}

    def _set_openai_env_configuration_vars(self) -> None:
        """Set unified environment variable names for both API versions."""
        dimensions = resolution_dimensions or self.DEFAULT_RESOLUTION_DIMENSIONS
        temp = dimensions.split("x")
        self._height = temp[1]
        self._width = temp[0]

        self._n_seconds = n_seconds or self.DEFAULT_N_SECONDS
        self._n_variants = n_variants or self.DEFAULT_N_VARIANTS
        self._api_version = api_version or self.DEFAULT_API_VERSION

        # Validate input based on resolution dimensions
        self._validate_video_constraints(
            resolution_dimensions=dimensions,
            n_seconds=self._n_seconds,
            n_variants=self._n_variants,
        )

        self._output_filename = output_filename

        self._params = {}
        self._params["api-version"] = self._api_version

        # Set expected routes for URL validation (Sora video generation API)
        self._expected_route = [
            "/videos/generations",
        ]
        # Validate endpoint URL
        self._warn_if_irregular_endpoint()

    def _set_openai_env_configuration_vars(self):
        self.model_name_environment_variable = "OPENAI_SORA_MODEL"
        self.endpoint_environment_variable = "OPENAI_SORA_ENDPOINT"
        self.api_key_environment_variable = "OPENAI_SORA_KEY"

    def _detect_api_version(self) -> str:
        """
        Detect the API version based on the endpoint URL.

        Returns:
            str: The detected API version ("v1" or "v2").
        """
        if "v1/videos" in self._endpoint:
            return "v2"
        else:
            return "v1"

    def _is_azure_openai(self) -> bool:
        """
        Detect whether this is an Azure OpenAI endpoint.

        Returns:
            bool: True if Azure OpenAI, False if OpenAI.
        """
        return "azure.com" in self._endpoint.lower() or "azure" in self._endpoint.lower()

    def _parse_and_validate_resolution(self, *, resolution_dimensions: str) -> tuple[str, str]:
        """
        Parse and validate the resolution dimensions string.

        Args:
            resolution_dimensions (str): Resolution dimensions in WIDTHxHEIGHT format.

        Returns:
            tuple[str, str]: A tuple of (width, height) as strings.

        Raises:
            ValueError: If the resolution format is invalid or unsupported for the detected API version.
        """
        # Parse resolution format
        if "x" not in resolution_dimensions:
            raise ValueError(
                f"Invalid resolution format: '{resolution_dimensions}'. "
                "Expected format: 'WIDTHxHEIGHT' (e.g., '1280x720')"
            )

        dimensions = resolution_dimensions.strip().lower().split("x")
        if len(dimensions) != 2:
            raise ValueError(
                f"Invalid resolution format: '{resolution_dimensions}'. "
                "Expected format: 'WIDTHxHEIGHT' (e.g., '1280x720')"
            )

        # Validate resolution based on detected API version
        if self._detected_api_version == "v1":
            if resolution_dimensions not in self.V1_RESOLUTIONS:
                endpoint_type = "Azure OpenAI" if self._is_azure_openai() else "OpenAI"
                logger.warning(
                    f"Resolution '{resolution_dimensions}' may not be supported for Sora-1 on {endpoint_type}. "
                    f"Commonly supported resolutions: {', '.join(self.V1_RESOLUTIONS)}. "
                    "The API may reject this request."
                )

            if resolution_dimensions in ["1080x1080", "1920x1080"] and self._n_seconds > 10:
                raise ValueError(
                    "n_seconds must be less than or equal to 10 for "
                    f"resolution dimensions of {resolution_dimensions}."
                )
        else:  # v2
            endpoint_type = "Azure OpenAI" if self._is_azure_openai() else "OpenAI"
            if resolution_dimensions not in self.V2_RESOLUTIONS:
                logger.warning(
                    f"Resolution '{resolution_dimensions}' may not be supported for Sora-2 on {endpoint_type}. "
                    f"Supported resolutions: {', '.join(self.V2_RESOLUTIONS)}. "
                    "The API may reject this request."
                )

        width = dimensions[0]
        height = dimensions[1]

        return width, height

    def _validate_duration(self) -> None:
        """
        Validate the video duration based on the detected API version.

        Raises:
            ValueError: If the duration is invalid for the detected API version.
        """
        # Basic duration validation
        if self._n_seconds <= 0:
            raise ValueError(f"Invalid duration: {self._n_seconds}. Duration must be greater than 0 seconds.")

        # API-specific duration validation
        if self._detected_api_version == "v1":
            if self._n_seconds > 20:
                raise ValueError(f"Invalid duration for Sora-1: {self._n_seconds}. Maximum duration is 20 seconds.")
        else:  # v2
            if self._n_seconds not in self.V2_DURATIONS:
                raise ValueError(
                    f"Invalid duration for Sora-2: {self._n_seconds}. "
                    f"Supported durations: {', '.join(map(str, self.V2_DURATIONS))} seconds."
                )

    async def _send_httpx_request_async(
        self,
        *,
        endpoint_uri: str,
        method: str,
        request_body: Optional[dict[str, object]] = None,
        files: Optional[dict[str, tuple]] = None,
    ) -> httpx.Response:
        """
        Asynchronously send an HTTP request using the httpx client and handle exceptions.

        Raises:
            RateLimitException: If the rate limit is exceeded.
            httpx.HTTPStatusError: If the request fails.
        """
        try:
            response = await net_utility.make_request_and_raise_if_error_async(
                endpoint_uri=endpoint_uri,
                method=method,
                request_body=request_body,
                files=files,
                headers=self._headers,
                params=self._params,
                **self._httpx_client_kwargs,
            )
        except httpx.HTTPStatusError as StatusError:
            logger.error(f"HTTP Status Error: {StatusError.response.status_code} - {StatusError.response.text}")
            if StatusError.response.status_code == 429:
                raise RateLimitException()
            else:
                raise

        return response

    @limit_requests_per_minute
    @pyrit_target_retry
    async def send_prompt_async(self, *, prompt_request: Message) -> Message:
        """Asynchronously sends a message and handles the response within a managed conversation context.

        Args:
            message (Message): The message object.

        Returns:
            Message: The updated conversation entry with the response from the prompt target.

        Raises:
            RateLimitException: If the rate limit is exceeded.
            httpx.HTTPStatusError: If the request fails.
        """
        self._validate_request(prompt_request=prompt_request)
        request = prompt_request.message_pieces[0]
        prompt = request.converted_value

        logger.info(f"Sending the following prompt to the prompt target: {prompt}")

        # Use the API version detected from the endpoint URL
        if self._detected_api_version == "v1":
            return await self._send_v1_request_async(request, prompt)
        else:
            return await self._send_v2_request_async(request, prompt)

    def _handle_http_error(self, error: httpx.HTTPStatusError, request: MessagePiece) -> Message:
        """
        Handle HTTP errors with standardized error parsing and response construction.

        Args:
            error: The HTTPStatusError to handle.
            request: The request piece associated with the prompt.

        Returns:
            Message: The error response entry.
        """
        # Extract error content from HTTP response
        try:
            error_dict = error.response.json()
        except Exception:
            error_dict = {"error": error.response.text}

        error_details, is_content_filter = self._extract_error_info(
            error_dict=error_dict,
            status_code=error.response.status_code,
        )

        error_message = "; ".join(error_details)
        logger.error(f"HTTP error during prompt send: {error_message}")

        if is_content_filter:
            return handle_bad_request_exception(
                response_text=error_message,
                request=request,
                is_content_filter=True,
            )
        else:
            # Determine error type based on error details
            error_type: PromptResponseError = "unknown"

            # Check for specific error types that indicate processing/request errors
            if any(
                err_type in error_message.lower()
                for err_type in [
                    "invalid_request_error",
                    "video_generation_user_error",
                    "invalid_value",
                    "user error",
                    "resolution",  # Handles "resolution not supported" errors from Sora-1
                    "not supported",
                ]
            ):
                error_type = "processing"

            return construct_response_from_request(
                request=request,
                response_text_pieces=[error_message],
                response_type="error",
                error=error_type,
            )

    async def _send_v1_request_async(self, request: MessagePiece, prompt: str) -> Message:
        """Send request using Sora-1 API (JSON body)."""
        body = self._construct_v1_request_body(prompt=prompt)
        endpoint_uri = f"{self._endpoint}/jobs"

        # Set api-version parameter for v1
        self._params["api-version"] = "preview"

        try:
            response = await self._send_httpx_request_async(
                endpoint_uri=endpoint_uri,
                method="POST",
                request_body=body,
            )
            return await self._handle_response_async(request=request, response=response)

        except httpx.HTTPStatusError as e:
            return self._handle_http_error(error=e, request=request)

    async def _send_v2_request_async(self, request: MessagePiece, prompt: str) -> Message:
        """Send request using Sora-2 API (multipart form data)."""
        files = self._construct_v2_request_files(prompt=prompt)

        # Remove api-version parameter for v2
        self._params.pop("api-version", None)

        try:
            response = await self._send_httpx_request_async(
                endpoint_uri=self._endpoint,
                method="POST",
                files=files,
            )
            return await self._handle_response_async(request=request, response=response)

        except httpx.HTTPStatusError as e:
            return self._handle_http_error(error=e, request=request)

    @pyrit_custom_result_retry(
        retry_function=_should_retry_check_job,
        retry_max_num_attempts=CHECK_JOB_RETRY_MAX_NUM_ATTEMPTS,
    )
    @pyrit_target_retry
    async def check_job_status_async(self, task_id: str) -> httpx.Response:
        """Check job/task status using the appropriate endpoint."""
        if self._detected_api_version == "v1":
            uri = f"{self._endpoint}/jobs/{task_id}"
        else:
            uri = f"{self._endpoint}/{task_id}"

        response = await self._send_httpx_request_async(
            endpoint_uri=uri,
            method="GET",
        )

        # Log the current status for visibility
        try:
            content = json.loads(response.content)
            status = content.get("status", "unknown")
            api_label = "Job" if self._detected_api_version == "v1" else "Task"
            logger.info(f"{api_label} {task_id} status: {status}")
        except Exception:
            pass

        return response

    @pyrit_custom_result_retry(
        retry_function=_should_retry_video_download,
    )
    @pyrit_target_retry
    async def download_video_content_async(
        self, task_id: str, generation_id: Optional[str] = None, **kwargs
    ) -> httpx.Response:
        """Download video using the appropriate endpoint."""
        if self._detected_api_version == "v1":
            if generation_id is None:
                raise ValueError("generation_id required for Sora v1 API")
            logger.info(f"Downloading video content for generation ID: {generation_id}")
            uri = f"{self._endpoint}/{generation_id}/content/video"
        else:
            logger.info(f"Downloading video content for task ID: {task_id}")
            uri = f"{self._endpoint}/{task_id}/content"

        # Use longer timeout for video download (2 minutes) for v2
        if self._detected_api_version == "v2":
            download_kwargs = self._httpx_client_kwargs.copy()
            download_kwargs["timeout"] = 120.0

            response = await net_utility.make_request_and_raise_if_error_async(
                endpoint_uri=uri,
                method="GET",
                headers=self._headers,
                params=self._params,
                **download_kwargs,
            )
        else:
            response = await self._send_httpx_request_async(
                endpoint_uri=uri,
                method="GET",
            )

        return response

    async def _handle_response_async(self, *, request: MessagePiece, response: httpx.Response) -> Message:
        """
        Asynchronously handle the response to a video generation request.

        This includes checking the status of the task and downloading the video content if successful.

        Args:
            request (MessagePiece): The message piece associated with the prompt.
            response (httpx.Response): The response from the API.

        Returns:
            Message: The response entry with the saved video path or error message.
        """
        content = json.loads(response.content)
        task_id = content.get("id")
        logger.info(f"Handling response for Task ID: {task_id}")

        # Check status with retry until task is complete
        try:
            task_response = await self.check_job_status_async(task_id=task_id)
            task_content = json.loads(task_response.content)
            status = task_content.get("status")
        except Exception as e:
            # If status check fails after all retries, log the full error details
            logger.error(f"Failed to check job status for task {task_id} after all retries. Error: {e}")
            error_msg = f"Video generation task {task_id} status check failed: {str(e)}"
            return construct_response_from_request(
                request=request,
                response_text_pieces=[error_msg],
                response_type="error",
                error="unknown",
            )

        # Handle task based on status
        if status in [JobStatus.SUCCEEDED.value, JobStatus.COMPLETED.value]:
            return await self._download_and_save_video_async(
                task_id=task_id,
                task_content=task_content,
                request=request,
            )
        elif status in [JobStatus.FAILED.value, JobStatus.CANCELLED.value]:
            return self._handle_failed_task(
                task_id=task_id,
                status=status,
                task_content=task_content,
                request=request,
            )
        else:
            return self._handle_processing_task(
                task_id=task_id,
                status=status,
                task_content=task_content,
                request=request,
            )

    def _handle_failed_task(
        self,
        *,
        task_id: str,
        status: str,
        task_content: dict,
        request: MessagePiece,
    ) -> Message:
        """
        Handle a failed or cancelled video generation task.

        Args:
            task_id (str): The ID of the failed task.
            status (str): The status value (failed or cancelled).
            task_content (dict): The response content from the status check.
            request (MessagePiece): The message piece associated with the prompt.

        Returns:
            Message: The error response entry.
        """
        failure_reason = task_content.get("failure_reason", None)
        error_details, is_content_filter = self._extract_error_info(
            error_dict=task_content,
            failure_reason=failure_reason,
        )

        failure_message = f"{task_id} {status}. {'; '.join(error_details)}"
        logger.error(failure_message)

        # Check if failure was due to content moderation
        if is_content_filter:
            return handle_bad_request_exception(
                response_text=failure_message,
                request=request,
                is_content_filter=True,
            )
        else:
            return construct_response_from_request(
                request=request,
                response_text_pieces=[failure_message],
                response_type="error",
                error="unknown",
            )

    def _extract_error_info(
        self,
        *,
        error_dict: dict,
        failure_reason: Optional[str] = None,
        status_code: Optional[int] = None,
    ) -> tuple[list[str], bool]:
        """
        Extract error information from error response or task content.

        Args:
            error_dict (dict): Dictionary containing error information.
            failure_reason (Optional[str]): Failure reason from task status.
            status_code (Optional[int]): HTTP status code if from HTTP error.

        Returns:
            tuple[list[str], bool]: (error_details, is_content_filter)
        """
        error_details = []
        is_content_filter = False

        # Add status code if present
        if status_code:
            error_details.append(f"HTTP {status_code}")

        # Add failure reason if present (from task status)
        if failure_reason:
            error_details.append(f"Failure reason: {failure_reason}")
            # Check if it's a moderation failure
            if failure_reason in [FailureReason.INPUT_MODERATION.value, FailureReason.OUTPUT_MODERATION.value]:
                is_content_filter = True

        # Extract error details from error dict
        if "error" in error_dict:
            error_info = error_dict["error"]
            if isinstance(error_info, dict):
                for key in ["message", "type", "code", "param"]:
                    if key in error_info:
                        error_details.append(f"{key.capitalize()}: {error_info[key]}")
            else:
                error_details.append(f"Error: {error_info}")

        # Fallback to raw content if no structured errors
        if not error_details or (len(error_details) == 1 and status_code):
            error_details.append(f"Raw response: {error_dict}")

        # Build error message for content filter keyword check
        error_message = "; ".join(error_details).lower()

        # Check for content filtering keywords (for HTTP errors)
        if status_code == 400 or not is_content_filter:
            is_content_filter = is_content_filter or any(
                keyword in error_message
                for keyword in ["content", "policy", "moderation", "filter", "inappropriate", "violation"]
            )

        return error_details, is_content_filter

    def _handle_processing_task(
        self,
        *,
        task_id: str,
        status: str,
        task_content: dict,
        request: MessagePiece,
    ) -> Message:
        """
        Handle a task that is still processing after max retries.

        Args:
            task_id (str): The ID of the processing task.
            status (str): The current status value.
            task_content (dict): The response content from the status check.
            request (MessagePiece): The message piece associated with the prompt.

        Returns:
            Message: The response entry with task status information.
        """
        logger.info(
            f"{task_id} is still processing after attempting retries. Consider setting a value > "
            + f"{self.CHECK_JOB_RETRY_MAX_NUM_ATTEMPTS} for environment variable "
            + f"CUSTOM_RESULT_RETRY_MAX_NUM_ATTEMPTS. Status: {status}"
        )
        return construct_response_from_request(
            request=request,
            response_text_pieces=[f"{str(task_content)}"],
        )

    async def _download_and_save_video_async(
        self,
        *,
        task_id: str,
        task_content: dict,
        request: MessagePiece,
    ) -> Message:
        """Download and save video using the appropriate method."""
        # Determine generation_id and file_name suffix based on API version
        if self._detected_api_version == "v1":
            generations = task_content.get("generations", [])
            generation_id = generations[0].get("id") if generations else None
            file_name_suffix = f"{task_id}_{generation_id}"
        else:
            generation_id = None
            file_name_suffix = f"{task_id}"

        # Download video content
        video_response = await self.download_video_content_async(task_id=task_id, generation_id=generation_id)

        # Use task/generation IDs as file name to ensure uniqueness
        return await self._save_video_to_storage_async(
            data=video_response.content,
            file_name=file_name_suffix,
            request=request,
        )

    async def _save_video_to_storage_async(
        self,
        *,
        data: bytes,
        file_name: str,
        request: MessagePiece,
    ) -> Message:
        """
        Asynchronously save the video content to storage using a serializer.

        Args:
            data (bytes): The video content to save.
            file_name (str): The filename to use.
            request (MessagePiece): The request piece associated with the prompt.

        Returns:
            Message: The response entry with the saved video path.
        """
        serializer = data_serializer_factory(
            category="prompt-memory-entries",
            data_type="video_path",
        )

        await serializer.save_data(data=data, output_filename=file_name)
        logger.info(f"Video response saved successfully to {serializer.value}")

        response_entry = construct_response_from_request(
            request=request, response_text_pieces=[str(serializer.value)], response_type="video_path"
        )

        return response_entry

    def _construct_v1_request_body(self, prompt: str) -> dict:
        """Constructs the JSON request body for Sora-1."""
        body_parameters: dict[str, object] = {
            "model": self._model_name,
            "prompt": prompt,
            "height": self._height,
            "width": self._width,
            "n_seconds": self._n_seconds,
        }
        return {k: v for k, v in body_parameters.items() if v is not None}

    def _construct_v2_request_files(self, prompt: str) -> dict:
        """Constructs the multipart form data files for Sora-2."""
        size = f"{self._width}x{self._height}"
        files_parameters: dict[str, tuple] = {
            "prompt": (None, prompt),
            "model": (None, self._model_name),
            "size": (None, size),
            "seconds": (None, str(self._n_seconds)),
        }
        return {k: v for k, v in files_parameters.items() if v[1] is not None}

    def _validate_request(self, *, prompt_request: Message) -> None:
        """
        Validates the message to ensure it meets the requirements for the Sora target.

        Args:
            prompt_request (Message): The message object.

        Raises:
            ValueError: If the request is invalid.
        """
        message_piece = prompt_request.get_piece()

        n_pieces = len(prompt_request.message_pieces)
        if n_pieces != 1:
            raise ValueError(f"This target only supports a single message piece. Received: {n_pieces} pieces.")

        piece_type = message_piece.converted_value_data_type
        if piece_type != "text":
            raise ValueError(f"This target only supports text prompt input. Received: {piece_type}.")

    def is_json_response_supported(self) -> bool:
        """Indicates that this target supports JSON response format."""
        return False
