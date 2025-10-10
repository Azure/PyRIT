# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import json
import logging
from typing import Literal, Optional

import httpx

from pyrit.common import net_utility
from pyrit.exceptions import handle_bad_request_exception, pyrit_custom_result_retry, pyrit_target_retry
from pyrit.models import PromptRequestPiece, PromptRequestResponse, construct_response_from_request
from pyrit.prompt_target import limit_requests_per_minute
from pyrit.prompt_target.openai.openai_sora_target_base import OpenAISoraTargetBase

logger = logging.getLogger(__name__)


class OpenAISora2Target(OpenAISoraTargetBase):
    """
    OpenAI Sora-2 Target class for the new OpenAI Sora API.
    Uses multipart form data and direct task endpoints with parameters like size, seconds.
    
    Supported resolutions: 720x1280, 1280x720, 1080x1920, 1920x1080
    Supported durations: 4, 8, 12 seconds
    """
    
    # Override default resolution for Sora-2
    DEFAULT_RESOLUTION_DIMENSIONS: str = "1280x720"
    
    # Override default duration for Sora-2 (only supports 4, 8, 12)
    DEFAULT_N_SECONDS: int = 4

    def __init__(
        self,
        *,
        resolution_dimensions: Optional[
            Literal["720x1280", "1280x720", "1080x1920", "1920x1080"]
        ] = None,
        n_seconds: Optional[Literal[4, 8, 12]] = None,
        output_filename: Optional[str] = None,
        **kwargs,
    ):
        """Initialize the OpenAI Sora-2 Target (new API).

        Args:
            model_name (str, Optional): The name of the model. Defaults to "sora-2".
                Can be set via OPENAI_SORA2_MODEL environment variable.
            endpoint (str, Optional): The target URL for the OpenAI service.
                Can be set via OPENAI_SORA2_ENDPOINT environment variable.
            api_key (str, Optional): The API key for accessing the service.
                Can be set via OPENAI_SORA2_KEY environment variable.
            headers (str, Optional): Extra headers of the endpoint (JSON).
                Can be set via OPENAI_ADDITIONAL_REQUEST_HEADERS environment variable.
            max_requests_per_minute (int, Optional): Number of requests the target can handle per
                minute before hitting a rate limit.
            httpx_client_kwargs (dict, Optional): Additional kwargs to be passed to the
                `httpx.AsyncClient()` constructor.
            resolution_dimensions (Literal, Optional): Resolution dimensions for the video.
                Defaults to "1280x720".
            n_seconds (Literal[4, 8, 12], Optional): The duration of the generated video (in seconds). 
                Must be 4, 8, or 12. Defaults to 4.
            output_filename (str, Optional): The name of the output file for the generated video.
                Note: DO NOT SET if using target with PromptSendingAttack. Default is {task_id}.mp4.

        Raises:
            ValueError: If n_seconds is not 4, 8, or 12, or if video constraints are not met.
        """
        # Set default n_seconds if not provided
        if n_seconds is None:
            n_seconds = 4  # Use the supported default value directly
            
        # Validate n_seconds for Sora-2 API
        if n_seconds not in [4, 8, 12]:
            raise ValueError(
                f"n_seconds must be 4, 8, or 12 for Sora-2 API. Got: {n_seconds}. "
                "Supported values are: 4, 8, and 12."
            )
        
        super().__init__(
            resolution_dimensions=resolution_dimensions,
            n_seconds=n_seconds,
            output_filename=output_filename,
            **kwargs,
        )

        # Validate resolution constraints for Sora-2
        dimensions = resolution_dimensions or self.DEFAULT_RESOLUTION_DIMENSIONS
        # Sora-2 has simpler constraints - all supported resolutions work with all supported durations
        # No additional validation needed beyond the n_seconds check above

        # No api-version parameter for Sora-2

    def _set_openai_env_configuration_vars(self) -> None:
        """Override to use OPENAI_SORA2_* environment variables."""
        self.model_name_environment_variable = "OPENAI_SORA2_MODEL"
        self.endpoint_environment_variable = "OPENAI_SORA2_ENDPOINT"
        self.api_key_environment_variable = "OPENAI_SORA2_KEY"

    @limit_requests_per_minute
    @pyrit_target_retry
    async def send_prompt_async(self, *, prompt_request: PromptRequestResponse) -> PromptRequestResponse:
        """Asynchronously sends a prompt request using multipart form data."""
        self._validate_request(prompt_request=prompt_request)
        request = prompt_request.request_pieces[0]
        prompt = request.converted_value

        logger.info(f"Sending the following prompt to the prompt target: {prompt}")

        files = self._construct_request_files(prompt=prompt)

        try:
            response = await self._send_httpx_request_async(
                endpoint_uri=self._endpoint,
                method="POST",
                files=files,
            )
            return await self._handle_response_async(request=request, response=response)
        
        except httpx.HTTPStatusError as e:
            # Handle HTTP errors (400, 403, etc.) and convert to proper response
            error_details = []
            error_details.append(f"HTTP {e.response.status_code}")
            
            try:
                # Try to parse JSON error response
                error_content = e.response.json()
                if "error" in error_content:
                    error_info = error_content["error"]
                    if isinstance(error_info, dict):
                        if "message" in error_info:
                            error_details.append(f"Message: {error_info['message']}")
                        if "type" in error_info:
                            error_details.append(f"Type: {error_info['type']}")
                        if "code" in error_info:
                            error_details.append(f"Code: {error_info['code']}")
                        if "param" in error_info:
                            error_details.append(f"Parameter: {error_info['param']}")
                    else:
                        error_details.append(f"Error: {error_info}")
                else:
                    error_details.append(f"Response: {error_content}")
            except Exception:
                # If JSON parsing fails, use raw text
                error_details.append(f"Raw response: {e.response.text}")
            
            error_message = "; ".join(error_details)
            logger.error(f"HTTP error during prompt send: {error_message}")
            
            # Check if it's a content filtering error (400 with moderation-related message)
            is_content_filter = (
                e.response.status_code == 400 and 
                any(keyword in error_message.lower() for keyword in 
                    ["content", "policy", "moderation", "filter", "inappropriate", "violation"])
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
        retry_function=OpenAISoraTargetBase._should_retry_check_job,
        retry_max_num_attempts=OpenAISoraTargetBase.CHECK_JOB_RETRY_MAX_NUM_ATTEMPTS,
    )
    @pyrit_target_retry
    async def check_job_status_async(self, task_id: str) -> httpx.Response:
        """Check task status using /{task_id} endpoint."""
        uri = f"{self._endpoint}/{task_id}"
        response = await self._send_httpx_request_async(
            endpoint_uri=uri,
            method="GET",
        )
        
        # Log the current status for visibility
        try:
            content = json.loads(response.content)
            status = content.get("status", "unknown")
            logger.info(f"Task {task_id} status: {status}")
        except Exception:
            pass
        
        return response

    # Removed problematic @pyrit_custom_result_retry decorator
    @pyrit_target_retry
    async def download_video_content_async(self, task_id: str, **kwargs) -> httpx.Response:
        """Download video using /{task_id}/content endpoint."""
        logger.info(f"Downloading video content for task ID: {task_id}")
        uri = f"{self._endpoint}/{task_id}/content"

        # Use longer timeout for video download (2 minutes)
        download_kwargs = self._httpx_client_kwargs.copy()
        download_kwargs['timeout'] = 120.0
        
        response = await net_utility.make_request_and_raise_if_error_async(
            endpoint_uri=uri,
            method="GET",
            headers=self._headers,
            params=self._params,
            **download_kwargs,
        )
        return response

    async def _download_and_save_video_async(
        self,
        *,
        task_id: str,
        task_content: dict,
        request: PromptRequestPiece,
    ) -> PromptRequestResponse:
        """Download and save video for Sora-2 API."""
        video_response = await self.download_video_content_async(task_id=task_id)

        file_name = self._output_filename if self._output_filename else f"{task_id}"

        return await self._save_video_to_storage_async(
            data=video_response.content,
            file_name=file_name,
            request=request,
        )

    def _construct_request_files(self, prompt: str) -> dict:
        """Constructs the multipart form data files for Sora-2."""
        size = f"{self._width}x{self._height}"

        files_parameters: dict[str, tuple] = {
            "prompt": (None, prompt),
            "model": (None, self._model_name),
            "size": (None, size),
            "seconds": (None, str(self._n_seconds)),
        }

        return {k: v for k, v in files_parameters.items() if v[1] is not None}
