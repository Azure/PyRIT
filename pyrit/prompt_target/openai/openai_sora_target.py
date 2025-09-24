# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import json
import logging
import os
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
    FAILED = "failed"
    CANCELLED = "cancelled"


class FailureReason(Enum):
    INPUT_MODERATION = "input_moderation"
    INTERNAL_ERROR = "internal_error"
    OUTPUT_MODERATION = "output_moderation"


class OpenAISoraTarget(OpenAITarget):
    """
    OpenAI Sora Target class for sending prompts to the OpenAI Sora API.
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
    DEFAULT_API_VERSION: str = "preview"
    DEFAULT_N_SECONDS: int = 5
    DEFAULT_N_VARIANTS: int = 1

    # Utility functions which define when to retry calls to check status and download video
    @staticmethod
    def _should_retry_check_job(response: httpx.Response) -> bool:
        """
        Returns True if the job status is not JobStatus.SUCCEEDED, JobStatus.FAILED, or JobStatus.CANCELLED.
        """
        content = json.loads(response.content)
        status = content.get("status", None)

        return status not in [JobStatus.SUCCEEDED.value, JobStatus.FAILED.value, JobStatus.CANCELLED.value]

    @staticmethod
    def _should_retry_video_download(response: httpx.Response) -> bool:
        """
        Returns True if the video download status is not 200 (success).
        """
        return response.status_code != 200

    def __init__(
        self,
        *,
        resolution_dimensions: Optional[
            Literal["360x360", "640x360", "480x480", "854x480", "720x720", "1280x720", "1080x1080", "1920x1080"]
        ] = None,
        n_seconds: Optional[int] = None,
        n_variants: Optional[int] = None,
        api_version: Optional[str] = None,
        output_filename: Optional[str] = None,
        **kwargs,
    ):
        """Initialize the OpenAI Sora Target.

        Args:
            model_name (str, Optional): The name of the model.
            endpoint (str, Optional): The target URL for the OpenAI service.
            api_key (str, Optional): The API key for accessing the Azure OpenAI service.
                Defaults to the `OPENAI_SORA_KEY` environment variable.
            headers (str, Optional): Extra headers of the endpoint (JSON).
            use_aad_auth (bool, Optional): When set to True, user authentication is used
                instead of API Key. DefaultAzureCredential is taken for
                https://cognitiveservices.azure.com/.default. Please run `az login` locally
                to leverage user AuthN.
            api_version (str, Optional): The version of the Azure OpenAI API. Defaults to
                "preview".
            max_requests_per_minute (int, Optional): Number of requests the target can handle per
                minute before hitting a rate limit. The number of requests sent to the target
                will be capped at the value provided.
            httpx_client_kwargs (dict, Optional): Additional kwargs to be passed to the
                `httpx.AsyncClient()` constructor.
            resolution_dimensions (Literal, Optional): Resolution dimensions for the video.
                Defaults to "480x480", where the first value is width and the second is height.
            n_seconds (int, Optional): The duration of the generated video (in seconds). Defaults to 5.
                Sora API will support duration up to 20s. For 1080p, maximum duration is 10s.
            n_variants (int, Optional): The number of generated videos. Defaults to 1. Sora API will support up
                to 2 variants for resolutions of 720p, but only 1 for resolutions of 1080p.
            output_filename (str, Optional): The name of the output file for the generated video.
                Note: DO NOT SET if using target with PromptSendingAttack. The default filename
                is {job_id}_{generation_id}.mp4 as returned by the model.

        Raises:
            ValueError: If video constraints are not met for the specified resolution.
        """
        super().__init__(**kwargs)

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
    async def send_prompt_async(self, *, prompt_request: PromptRequestResponse) -> PromptRequestResponse:
        """Asynchronously sends a prompt request and handles the response within a managed conversation context.

        Args:
            prompt_request (PromptRequestResponse): The prompt request response object.

        Returns:
            PromptRequestResponse: The updated conversation entry with the response from the prompt target.

        Raises:
            RateLimitException: If the rate limit is exceeded.
            httpx.HTTPStatusError: If the request fails.
        """
        self._validate_request(prompt_request=prompt_request)
        request = prompt_request.request_pieces[0]
        prompt = request.converted_value

        logger.info(f"Sending the following prompt to the prompt target: {prompt}")

        body = self._construct_request_body(prompt=prompt)

        response = await self._send_httpx_request_async(
            endpoint_uri=f"{self._endpoint}/jobs",
            method="POST",
            request_body=body,
        )

        return await self._handle_response_async(request=request, response=response)

    @pyrit_custom_result_retry(
        retry_function=_should_retry_check_job,
        retry_max_num_attempts=CHECK_JOB_RETRY_MAX_NUM_ATTEMPTS,
    )
    @pyrit_target_retry
    async def check_job_status_async(self, job_id: str) -> httpx.Response:
        """
        Asynchronously check status of a submitted video generation job using the job_id.

        Retries a maxium of {self.CHECK_JOB_RETRY_MAX_NUM_ATTEMPTS} times,
        until the job is complete (succeeded, failed, or cancelled). Also
        retries upon RateLimitException.

        Args:
            job_id (str): The ID of the job to check.

        Returns:
            httpx.Response: The response from the API.

        Raises:
            RateLimitException: If the rate limit is exceeded.
            httpx.HTTPStatusError: If the request fails.
        """

        uri = f"{self._endpoint}/jobs/{job_id}"

        response = await self._send_httpx_request_async(
            endpoint_uri=uri,
            method="GET",
        )

        return response

    @pyrit_custom_result_retry(
        retry_function=_should_retry_video_download,
    )
    @pyrit_target_retry
    async def download_video_content_async(self, generation_id: str) -> httpx.Response:
        """
        Asynchronously download the video using the video generation ID.

        Retries if the response status code is not 200. Also retries upon RateLimitException.

        Args:
            generation_id (str): The ID of the video generation to download.

        Returns:
            httpx.Response: The response from the API.

        Raises:
            RateLimitException: If the rate limit is exceeded.
            httpx.HTTPStatusError: If the request fails.
        """

        logger.info(f"Downloading video content for video generation ID: {generation_id}")
        uri = f"{self._endpoint}/{generation_id}/content/video"

        response = await self._send_httpx_request_async(
            endpoint_uri=uri,
            method="GET",
        )

        return response

    async def _save_video_to_storage_async(
        self,
        *,
        data: bytes,
        job_id: str,
        generation_id: str,
        request: PromptRequestPiece,
    ) -> PromptRequestResponse:
        """
        Asynchronously save the video content to storage using a serializer.

        This function is called after the video content is available for download.

        Args:
            data (bytes): The video content to save.
            job_id (str): The video generation job ID.
            generation_id (str): The video generation ID.
            request (PromptRequestPiece): The request piece associated with the prompt.

        Returns:
            PromptRequestResponse: The response entry with the saved video path.
        """
        serializer = data_serializer_factory(
            category="prompt-memory-entries",
            data_type="video_path",
        )
        file_name = self._output_filename if self._output_filename else f"{job_id}_{generation_id}"
        await serializer.save_data(data=data, output_filename=file_name)
        logger.info(f"Video response saved successfully to {serializer.value}")

        response_entry = construct_response_from_request(
            request=request, response_text_pieces=[str(serializer.value)], response_type="video_path"
        )

        return response_entry

    async def _handle_response_async(
        self, request: PromptRequestPiece, response: httpx.Response
    ) -> PromptRequestResponse:
        """
        Asynchronously handle the response to a video generation request.

        This includes checking the status of the job and downloading the video content if successful.

        Args:
            request (PromptRequestPiece): The request piece associated with the prompt.
            response (httpx.Response): The response from the API.

        Returns:
            PromptRequestResponse: The response entry with the saved video path or error message.
        """
        content = json.loads(response.content)

        job_id = content.get("id")
        logger.info(f"Handling response for Job ID: {job_id}")

        # Check status with retry until job is complete
        job_response = await self.check_job_status_async(job_id=job_id)
        job_content = json.loads(job_response.content)
        status = job_content.get("status")

        # Handle completed job
        if status == JobStatus.SUCCEEDED.value:
            # Download video content
            generations = job_content.get("generations")
            generation_id = generations[0].get("id")

            video_response = await self.download_video_content_async(generation_id=generation_id)
            response_entry = await self._save_video_to_storage_async(
                data=video_response.content,
                job_id=job_id,
                generation_id=generation_id,
                request=request,
            )
        elif status in [JobStatus.FAILED.value, JobStatus.CANCELLED.value]:
            failure_reason = job_content.get("failure_reason", None)
            failure_message = f"{job_id} {status}, Reason: {failure_reason}"
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
                f"{job_id} is still processing after attempting retries. Consider setting a value > "
                + f"{self.CHECK_JOB_RETRY_MAX_NUM_ATTEMPTS} for environment variable "
                + f"CUSTOM_RESULT_RETRY_MAX_NUM_ATTEMPTS. Status: {status}"
            )
            response_entry = construct_response_from_request(
                request=request,
                response_text_pieces=[f"{str(job_content)}"],
            )

        return response_entry

    def _validate_video_constraints(self, resolution_dimensions: str, n_variants: int, n_seconds: int) -> None:
        """
        Validate the video constraints based on the resolution dimensions.

        This checks both n_seconds and n_variants values, which have different constraints for different resolution
        dimensions.

        Raises:
            ValueError: If the constraints are not met.
        """
        if resolution_dimensions in ["1080x1080", "1920x1080"]:
            # Constraints apply to all 1080p dimensions
            if n_seconds > 10:
                raise ValueError(
                    f"n_seconds must be less than or equal to 10 for resolution dimensions of {resolution_dimensions}."
                )

            if n_variants > 1:
                raise ValueError(
                    f"n_variants must be less than or equal to 1 for resolution dimensions of {resolution_dimensions}."
                )
        elif resolution_dimensions in ["720x720", "1280x720"]:
            # Constraints apply to all 720p dimensions
            if n_variants > 2:
                raise ValueError(
                    f"n_variants must be less than or equal to 2 for resolution dimensions of {resolution_dimensions}."
                )

        # Constraints apply for all dimensions outside of 1080p
        if n_seconds > 20 and resolution_dimensions not in ["1080x1080", "1920x1080"]:
            raise ValueError(
                f"n_seconds must be less than or equal to 20 for resolution dimensions of {resolution_dimensions}."
            )

    def _construct_request_body(self, prompt: str) -> dict:
        """
        Constructs the request body for the endpoint API.

        Args:
            prompt (str): The prompt text to be sent to the API.

        Returns:
            dict: The request body as a dictionary.
        """
        body_parameters: dict[str, object] = {
            "model": self._model_name,
            "prompt": prompt,
            "height": self._height,
            "width": self._width,
            "n_seconds": self._n_seconds,
            "n_variants": self._n_variants,
        }

        # Filter out None values
        return {k: v for k, v in body_parameters.items() if v is not None}

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

        messages = self._memory.get_chat_messages_with_conversation_id(conversation_id=request_piece.conversation_id)
        n_messages = len(messages)

        if n_messages > 0:
            raise ValueError(
                "This target only supports a single turn conversation. "
                f"Received: {n_messages} messages which indicates a prior turn."
            )

    def is_json_response_supported(self) -> bool:
        """Indicates that this target supports JSON response format."""
        return False
