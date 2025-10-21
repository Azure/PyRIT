# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Unified OpenAI Sora Target supporting both Sora-1 and Sora-2 APIs.

This module provides a single interface for both legacy and new Sora APIs.
The API version is automatically detected based on the endpoint response format.

Both APIs use unified environment variables:
- OPENAI_SORA_ENDPOINT
- OPENAI_SORA_KEY
- OPENAI_SORA_MODEL
"""

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
    construct_response_from_request,
    data_serializer_factory,
)
from pyrit.prompt_target import OpenAITarget, limit_requests_per_minute

logger = logging.getLogger(__name__)


class JobStatus(Enum):
    PREPROCESSING = "preprocessing"
    QUEUED = "queued"
    PROCESSING = "processing"
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
    - Sora-2: Endpoint contains "openai/v1/videos"
    - Sora-1: All other endpoints

    Sora-1 API:
    - Uses JSON body with /jobs endpoints
    - Supported resolutions: 360x360, 640x360, 480x480, 854x480, 720x720, 1280x720, 1080x1080, 1920x1080
    - Duration: up to 20s (10s for 1080p)

    Sora-2 API:
    - Uses multipart form data with direct task endpoints
    - Supported resolutions: 720x1280, 1280x720, 1080x1920, 1920x1080
    - Duration: 4, 8, or 12 seconds only

    Default resolution (1280x720) works with both APIs.
    Default duration (4s) works with both APIs (v1 supports up to 20s, v2 requires 4/8/12s).
    """

    # Maximum number of retries for check_job_status_async()
    CHECK_JOB_RETRY_MAX_NUM_ATTEMPTS: int = int(os.getenv("CUSTOM_RESULT_RETRY_MAX_NUM_ATTEMPTS", "60"))

    # Resolution sets
    V1_RESOLUTIONS = ["360x360", "640x360", "480x480", "854x480", "720x720", "1280x720", "1080x1080", "1920x1080"]
    V2_RESOLUTIONS = ["720x1280", "1280x720", "1080x1920", "1920x1080"]
    V2_DURATIONS = [4, 8, 12]

    # Utility functions which define when to retry calls to check status and download video
    @staticmethod
    def _should_retry_check_job(response: httpx.Response) -> bool:
        """
        Returns True if the job status is not SUCCEEDED, COMPLETED, FAILED, or CANCELLED.
        """
        content = json.loads(response.content)
        status = content.get("status", None)
        return status not in [
            JobStatus.SUCCEEDED.value,
            JobStatus.COMPLETED.value,
            JobStatus.FAILED.value,
            JobStatus.CANCELLED.value,
        ]

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
        n_variants: int = 1,
        output_filename: Optional[str] = None,
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
                Defaults to "1280x720" which works with both API versions.
                Sora-1 supported resolutions are "360x360", "640x360", "480x480", "854x480",
                "720x720", "1280x720", "1080x1080", "1920x1080".
                Sora-2 supported resolutions are "720x1280", "1280x720", "1080x1920", "1920x1080".
            n_seconds (int, Optional): The duration of the generated video (in seconds).
                Defaults to 4 (compatible with both APIs).
                Sora-1 supports up to 20 seconds (10 seconds max for 1080p resolutions).
                Sora-2 supports exactly 4, 8, or 12 seconds.
            n_variants (int, Optional): Number of video variants to generate. Defaults to 1.
                Only supported by Sora-1 API (ignored for Sora-2).
            output_filename (str, Optional): The name of the output file for the generated video.
                Note: DO NOT SET if using target with PromptSendingAttack.
        """
        # Parse resolution
        if "x" not in resolution_dimensions:
            raise ValueError(
                f"Invalid resolution format: '{resolution_dimensions}'. "
                "Expected format: 'WIDTHxHEIGHT' (e.g., '1280x720')"
            )
        dimensions = resolution_dimensions.split("x")
        if len(dimensions) != 2:
            raise ValueError(
                f"Invalid resolution format: '{resolution_dimensions}'. "
                "Expected format: 'WIDTHxHEIGHT' (e.g., '1280x720')"
            )
        self._height = dimensions[1]
        self._width = dimensions[0]

        self._n_seconds = n_seconds
        self._n_variants = n_variants
        self._output_filename = output_filename
        self._params: Dict[str, Any] = {}  # Initialize params dict

        # Initialize parent class first to get endpoint
        super().__init__(**kwargs)

        # Detect API version based on endpoint URL
        if "openai/v1/videos" in self._endpoint:
            self._detected_api_version = "v2"
        else:
            self._detected_api_version = "v1"

    def _set_openai_env_configuration_vars(self) -> None:
        """Set unified environment variable names for both API versions."""
        self.model_name_environment_variable = "OPENAI_SORA_MODEL"
        self.endpoint_environment_variable = "OPENAI_SORA_ENDPOINT"
        self.api_key_environment_variable = "OPENAI_SORA_KEY"

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
    async def send_prompt_async(self, *, message: Message) -> Message:
        """Asynchronously sends a message and handles the response within a managed conversation context.

        Args:
            message (Message): The message object.

        Returns:
            Message: The updated conversation entry with the response from the prompt target.

        Raises:
            RateLimitException: If the rate limit is exceeded.
            httpx.HTTPStatusError: If the request fails.
        """
        self._validate_request(message=message)
        request = message.message_pieces[0]
        prompt = request.converted_value

        logger.info(f"Sending the following prompt to the prompt target: {prompt}")

        # Use the API version detected from the endpoint URL
        if self._detected_api_version == "v1":
            return await self._send_v1_request_async(request, prompt)
        else:
            return await self._send_v2_request_async(request, prompt)

    async def _send_v1_request_async(self, request: MessagePiece, prompt: str) -> Message:
        """Send request using Sora-1 API (JSON body)."""
        body = self._construct_v1_request_body(prompt=prompt)
        endpoint_uri = f"{self._endpoint}/jobs"

        # Set api-version parameter for v1
        self._params["api-version"] = "preview"

        response = await self._send_httpx_request_async(
            endpoint_uri=endpoint_uri,
            method="POST",
            request_body=body,
        )

        self._detected_api_version = "v1"
        return await self._handle_response_async(request=request, response=response)

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
            self._detected_api_version = "v2"
            return await self._handle_response_async(request=request, response=response)

        except httpx.HTTPStatusError as e:
            # Handle HTTP errors for v2
            error_details = [f"HTTP {e.response.status_code}"]

            try:
                error_content = e.response.json()
                if "error" in error_content:
                    error_info = error_content["error"]
                    if isinstance(error_info, dict):
                        for key in ["message", "type", "code", "param"]:
                            if key in error_info:
                                error_details.append(f"{key.capitalize()}: {error_info[key]}")
                    else:
                        error_details.append(f"Error: {error_info}")
                else:
                    error_details.append(f"Response: {error_content}")
            except Exception:
                error_details.append(f"Raw response: {e.response.text}")

            error_message = "; ".join(error_details)
            logger.error(f"HTTP error during prompt send: {error_message}")

            # Check if it's a content filtering error
            is_content_filter = e.response.status_code == 400 and any(
                keyword in error_message.lower()
                for keyword in ["content", "policy", "moderation", "filter", "inappropriate", "violation"]
            )

            if is_content_filter:
                return handle_bad_request_exception(
                    response_text=error_message,
                    request=request,
                    is_content_filter=True,
                )
            else:
                return construct_response_from_request(
                    request=request,
                    response_text_pieces=[error_message],
                    response_type="error",
                    error="unknown",
                )

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

    async def _handle_response_async(
        self, request: MessagePiece, response: httpx.Response
    ) -> Message:
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
        task_response = await self.check_job_status_async(task_id=task_id)
        task_content = json.loads(task_response.content)
        status = task_content.get("status")

        # Handle completed task
        if status in [JobStatus.SUCCEEDED.value, JobStatus.COMPLETED.value]:
            response_entry = await self._download_and_save_video_async(
                task_id=task_id,
                task_content=task_content,
                request=request,
            )
        elif status in [JobStatus.FAILED.value, JobStatus.CANCELLED.value]:
            failure_reason = task_content.get("failure_reason", None)

            # Extract additional error details from the response
            error_details = []
            if failure_reason:
                error_details.append(f"Failure reason: {failure_reason}")

            # Look for additional error information in the response
            if "error" in task_content:
                error_info = task_content["error"]
                if isinstance(error_info, dict):
                    if "message" in error_info:
                        error_details.append(f"Error message: {error_info['message']}")
                    if "type" in error_info:
                        error_details.append(f"Error type: {error_info['type']}")
                    if "code" in error_info:
                        error_details.append(f"Error code: {error_info['code']}")
                else:
                    error_details.append(f"Error: {error_info}")

            # Include raw task content for debugging if no specific errors found
            if not error_details:
                error_details.append(f"Raw response: {task_content}")

            failure_message = f"{task_id} {status}. {'; '.join(error_details)}"
            logger.error(failure_message)

            if failure_reason in [FailureReason.INPUT_MODERATION.value, FailureReason.OUTPUT_MODERATION.value]:
                response_entry = handle_bad_request_exception(
                    response_text=failure_message,
                    request=request,
                    is_content_filter=True,
                )
            else:
                # Non-moderation failure reason
                response_entry = construct_response_from_request(
                    request=request,
                    response_text_pieces=[failure_message],
                    response_type="error",
                    error="unknown",
                )
        else:
            # Retry stop condition reached, return result
            logger.info(
                f"{task_id} is still processing after attempting retries. Consider setting a value > "
                + f"{self.CHECK_JOB_RETRY_MAX_NUM_ATTEMPTS} for environment variable "
                + f"CUSTOM_RESULT_RETRY_MAX_NUM_ATTEMPTS. Status: {status}"
            )
            response_entry = construct_response_from_request(
                request=request,
                response_text_pieces=[f"{str(task_content)}"],
            )

        return response_entry

    async def _download_and_save_video_async(
        self,
        *,
        task_id: str,
        task_content: dict,
        request: MessagePiece,
    ) -> Message:
        """Download and save video using the appropriate method."""
        if self._detected_api_version == "v1":
            generations = task_content.get("generations", [])
            generation_id = generations[0].get("id") if generations else None
            video_response = await self.download_video_content_async(task_id=task_id, generation_id=generation_id)
            file_name = self._output_filename if self._output_filename else f"{task_id}_{generation_id}"
        else:
            video_response = await self.download_video_content_async(task_id=task_id)
            file_name = self._output_filename if self._output_filename else f"{task_id}"

        return await self._save_video_to_storage_async(
            data=video_response.content,
            file_name=file_name,
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
            "n_variants": self._n_variants,
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

    def _validate_request(self, *, message: Message) -> None:
        """
        Validates the message to ensure it meets the requirements for the Sora target.

        Args:
            message (Message): The message object.

        Raises:
            ValueError: If the request is invalid.
        """
        message_piece = message.get_piece()

        n_pieces = len(message.message_pieces)
        if n_pieces != 1:
            raise ValueError(f"This target only supports a single message piece. Received: {n_pieces} pieces.")

        piece_type = message_piece.converted_value_data_type
        if piece_type != "text":
            raise ValueError(f"This target only supports text prompt input. Received: {piece_type}.")

    def is_json_response_supported(self) -> bool:
        """Indicates that this target supports JSON response format."""
        return False
