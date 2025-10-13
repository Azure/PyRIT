# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import json
import logging
import os
from abc import abstractmethod
from enum import Enum
from typing import Literal, Optional

import httpx

from pyrit.common import net_utility
from pyrit.exceptions import (
    RateLimitException,
    handle_bad_request_exception,
    pyrit_custom_result_retry,
    pyrit_target_retry,
)
from pyrit.models import (
    PromptRequestPiece,
    PromptRequestResponse,
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


class OpenAISoraTargetBase(OpenAITarget):
    """
    Base class for OpenAI Sora Target implementations.
    Contains common functionality shared between Sora-1 and Sora-2 APIs.
    """

    # Maximum number of retries for check_job_status_async()
    # This cannot be set in the constructor as it is used in the decorator, which does not know self.
    try:
        CHECK_JOB_RETRY_MAX_NUM_ATTEMPTS: int = int(os.getenv("CUSTOM_RESULT_RETRY_MAX_NUM_ATTEMPTS", 25))
    except ValueError:
        CHECK_JOB_RETRY_MAX_NUM_ATTEMPTS = 25
        logger.warning(
            "Invalid value for CUSTOM_RESULT_RETRY_MAX_NUM_ATTEMPTS. "
            + f"Using default value of {CHECK_JOB_RETRY_MAX_NUM_ATTEMPTS}."
        )

    DEFAULT_RESOLUTION_DIMENSIONS: str = "480x480"
    DEFAULT_N_SECONDS: int = 5

    # Utility functions which define when to retry calls to check status and download video
    @staticmethod
    def _should_retry_check_job(response: httpx.Response) -> bool:
        """
        Returns True if the job status is not JobStatus.SUCCEEDED, JobStatus.COMPLETED, JobStatus.FAILED, or JobStatus.CANCELLED.
        """
        content = json.loads(response.content)
        status = content.get("status", None)

        return status not in [JobStatus.SUCCEEDED.value, JobStatus.COMPLETED.value, JobStatus.FAILED.value, JobStatus.CANCELLED.value]

    @staticmethod
    def _should_retry_video_download(response: httpx.Response) -> bool:
        """
        Returns True if the video download status is not 200 (success).
        """
        return response.status_code != 200

    def __init__(
        self,
        *,
        resolution_dimensions: Optional[str] = None,
        n_seconds: Optional[int] = None,
        output_filename: Optional[str] = None,
        **kwargs,
    ):
        """Initialize the base OpenAI Sora Target.

        Args:
            model_name (str, Optional): The name of the model.
            endpoint (str, Optional): The target URL for the OpenAI service.
            api_key (str, Optional): The API key for accessing the service.
                Defaults to the `OPENAI_SORA_KEY` environment variable.
            headers (str, Optional): Extra headers of the endpoint (JSON).
            use_aad_auth (bool, Optional): When set to True, user authentication is used
                instead of API Key. DefaultAzureCredential is taken for
                https://cognitiveservices.azure.com/.default. Please run `az login` locally
                to leverage user AuthN.
            max_requests_per_minute (int, Optional): Number of requests the target can handle per
                minute before hitting a rate limit. The number of requests sent to the target
                will be capped at the value provided.
            httpx_client_kwargs (dict, Optional): Additional kwargs to be passed to the
                `httpx.AsyncClient()` constructor.
            resolution_dimensions (Literal, Optional): Resolution dimensions for the video.
                Defaults to "480x480", where the first value is width and the second is height.
            n_seconds (int, Optional): The duration of the generated video (in seconds). Defaults to 5.
                Sora API will support duration up to 20s. For 1080p, maximum duration is 10s.
            output_filename (str, Optional): The name of the output file for the generated video.
                Note: DO NOT SET if using target with PromptSendingAttack.

        Raises:
            ValueError: If video constraints are not met for the specified resolution.
        """
        super().__init__(**kwargs)

        dimensions = resolution_dimensions or self.DEFAULT_RESOLUTION_DIMENSIONS
        temp = dimensions.split("x")
        self._height = temp[1]
        self._width = temp[0]

        self._n_seconds = n_seconds or self.DEFAULT_N_SECONDS
        self._output_filename = output_filename
        self._params = {}  # Initialize params dict

    def _set_openai_env_configuration_vars(self):
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

    @abstractmethod
    @limit_requests_per_minute
    @pyrit_target_retry
    async def send_prompt_async(self, *, prompt_request: PromptRequestResponse) -> PromptRequestResponse:
        """Asynchronously sends a prompt request and handles the response within a managed conversation context."""
        pass

    @abstractmethod
    @pyrit_custom_result_retry(
        retry_function=_should_retry_check_job,
        retry_max_num_attempts=CHECK_JOB_RETRY_MAX_NUM_ATTEMPTS,
    )
    @pyrit_target_retry
    async def check_job_status_async(self, task_id: str) -> httpx.Response:
        """Asynchronously check status of a submitted video generation task."""
        pass

    @abstractmethod
    @pyrit_custom_result_retry(
        retry_function=_should_retry_video_download,
    )
    @pyrit_target_retry
    async def download_video_content_async(self, task_id: str, **kwargs) -> httpx.Response:
        """Asynchronously download the video content."""
        pass

    async def _handle_response_async(
        self, request: PromptRequestPiece, response: httpx.Response
    ) -> PromptRequestResponse:
        """
        Asynchronously handle the response to a video generation request.

        This includes checking the status of the task and downloading the video content if successful.

        Args:
            request (PromptRequestPiece): The request piece associated with the prompt.
            response (httpx.Response): The response from the API.

        Returns:
            PromptRequestResponse: The response entry with the saved video path or error message.
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

    @abstractmethod
    async def _download_and_save_video_async(
        self,
        *,
        task_id: str,
        task_content: dict,
        request: PromptRequestPiece,
    ) -> PromptRequestResponse:
        """Download and save video. Implementation differs between Sora-1 and Sora-2."""
        pass

    async def _save_video_to_storage_async(
        self,
        *,
        data: bytes,
        file_name: str,
        request: PromptRequestPiece,
    ) -> PromptRequestResponse:
        """
        Asynchronously save the video content to storage using a serializer.

        Args:
            data (bytes): The video content to save.
            file_name (str): The filename to use.
            request (PromptRequestPiece): The request piece associated with the prompt.

        Returns:
            PromptRequestResponse: The response entry with the saved video path.
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

    def _validate_request(self, *, prompt_request: PromptRequestResponse) -> None:
        """
        Validates the prompt request to ensure it meets the requirements for the Sora target.

        Args:
            prompt_request (PromptRequestResponse): The prompt request response object.

        Raises:
            ValueError: If the request is invalid.
        """
        request_piece = prompt_request.get_piece()

        n_pieces = len(prompt_request.request_pieces)
        if n_pieces != 1:
            raise ValueError(f"This target only supports a single prompt request piece. Received: {n_pieces} pieces.")

        piece_type = request_piece.converted_value_data_type
        if piece_type != "text":
            raise ValueError(f"This target only supports text prompt input. Received: {piece_type}.")

    def is_json_response_supported(self) -> bool:
        """Indicates that this target supports JSON response format."""
        return False
