# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import logging
from typing import Literal, Optional

import httpx

from pyrit.exceptions import pyrit_custom_result_retry, pyrit_target_retry
from pyrit.models import PromptRequestPiece, PromptRequestResponse
from pyrit.prompt_target import limit_requests_per_minute
from pyrit.prompt_target.openai.openai_sora_target_base import OpenAISoraTargetBase

logger = logging.getLogger(__name__)


class OpenAISora1Target(OpenAISoraTargetBase):
    """
    OpenAI Sora-1 Target class for the legacy Azure/OpenAI Sora API.
    Uses JSON body and /jobs endpoints with parameters like height, width, n_seconds, n_variants.
    """

    DEFAULT_API_VERSION: str = "preview"
    DEFAULT_N_VARIANTS: int = 1

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
        """Initialize the OpenAI Sora-1 Target (legacy API).

        Args:
            model_name (str, Optional): The name of the model.
            endpoint (str, Optional): The target URL for the OpenAI service.
            api_key (str, Optional): The API key for accessing the service.
                Defaults to the `OPENAI_SORA_KEY` environment variable.
            headers (str, Optional): Extra headers of the endpoint (JSON).
            use_aad_auth (bool, Optional): When set to True, user authentication is used
                instead of API Key.
            api_version (str, Optional): The version of the Azure OpenAI API. Defaults to "preview".
            max_requests_per_minute (int, Optional): Number of requests the target can handle per
                minute before hitting a rate limit.
            httpx_client_kwargs (dict, Optional): Additional kwargs to be passed to the
                `httpx.AsyncClient()` constructor.
            resolution_dimensions (Literal, Optional): Resolution dimensions for the video.
                Defaults to "480x480".
            n_seconds (int, Optional): The duration of the generated video (in seconds). Defaults to 5.
            n_variants (int, Optional): The number of generated videos. Defaults to 1.
            output_filename (str, Optional): The name of the output file for the generated video.
                Note: DO NOT SET if using target with PromptSendingAttack. Default is {job_id}_{generation_id}.mp4.

        Raises:
            ValueError: If video constraints are not met for the specified resolution.
        """
        super().__init__(
            resolution_dimensions=resolution_dimensions,
            n_seconds=n_seconds,
            output_filename=output_filename,
            **kwargs,
        )

        self._n_variants = n_variants or self.DEFAULT_N_VARIANTS
        self._api_version = api_version or self.DEFAULT_API_VERSION

        # Validate input based on resolution dimensions
        self._validate_video_constraints(
            resolution_dimensions=resolution_dimensions or self.DEFAULT_RESOLUTION_DIMENSIONS,
            n_seconds=self._n_seconds,
            n_variants=self._n_variants,
        )

        self._params["api-version"] = self._api_version

    def _validate_video_constraints(self, resolution_dimensions: str, n_variants: int, n_seconds: int) -> None:
        """
        Validate the video constraints based on the resolution dimensions.

        Raises:
            ValueError: If the constraints are not met.
        """
        if resolution_dimensions in ["1080x1080", "1920x1080"]:
            if n_seconds > 10:
                raise ValueError(
                    f"n_seconds must be less than or equal to 10 for resolution dimensions of {resolution_dimensions}."
                )
            if n_variants > 1:
                raise ValueError(
                    f"n_variants must be less than or equal to 1 for resolution dimensions of {resolution_dimensions}."
                )
        elif resolution_dimensions in ["720x720", "1280x720"]:
            if n_variants > 2:
                raise ValueError(
                    f"n_variants must be less than or equal to 2 for resolution dimensions of {resolution_dimensions}."
                )

        if n_seconds > 20 and resolution_dimensions not in ["1080x1080", "1920x1080"]:
            raise ValueError(
                f"n_seconds must be less than or equal to 20 for resolution dimensions of {resolution_dimensions}."
            )

    @limit_requests_per_minute
    @pyrit_target_retry
    async def send_prompt_async(self, *, prompt_request: PromptRequestResponse) -> PromptRequestResponse:
        """Asynchronously sends a prompt request using JSON body."""
        self._validate_request(prompt_request=prompt_request)
        request = prompt_request.request_pieces[0]
        prompt = request.converted_value

        logger.info(f"Sending the following prompt to the prompt target: {prompt}")

        body = self._construct_request_body(prompt=prompt)
        endpoint_uri = f"{self._endpoint}/jobs"

        response = await self._send_httpx_request_async(
            endpoint_uri=endpoint_uri,
            method="POST",
            request_body=body,
        )

        return await self._handle_response_async(request=request, response=response)

    @pyrit_custom_result_retry(
        retry_function=OpenAISoraTargetBase._should_retry_check_job,
        retry_max_num_attempts=OpenAISoraTargetBase.CHECK_JOB_RETRY_MAX_NUM_ATTEMPTS,
    )
    @pyrit_target_retry
    async def check_job_status_async(self, task_id: str) -> httpx.Response:
        """Check job status using /jobs/{job_id} endpoint."""
        uri = f"{self._endpoint}/jobs/{task_id}"
        response = await self._send_httpx_request_async(
            endpoint_uri=uri,
            method="GET",
        )
        
        # Log the current status for visibility
        try:
            import json
            content = json.loads(response.content)
            status = content.get("status", "unknown")
            logger.info(f"Job {task_id} status: {status}")
        except Exception:
            pass
        
        return response

    @pyrit_custom_result_retry(
        retry_function=OpenAISoraTargetBase._should_retry_video_download,
    )
    @pyrit_target_retry
    async def download_video_content_async(self, task_id: str, generation_id: str, **kwargs) -> httpx.Response:
        """Download video using /{generation_id}/content/video endpoint."""
        logger.info(f"Downloading video content for generation ID: {generation_id}")
        uri = f"{self._endpoint}/{generation_id}/content/video"

        response = await self._send_httpx_request_async(
            endpoint_uri=uri,
            method="GET",
        )
        return response

    async def _download_and_save_video_async(
        self,
        *,
        task_id: str,
        task_content: dict,
        request: PromptRequestPiece,
    ) -> PromptRequestResponse:
        """Download and save video for Sora-1 API."""
        generations = task_content.get("generations", [])
        generation_id = generations[0].get("id") if generations else None

        video_response = await self.download_video_content_async(task_id=task_id, generation_id=generation_id)

        file_name = self._output_filename if self._output_filename else f"{task_id}_{generation_id}"

        return await self._save_video_to_storage_async(
            data=video_response.content,
            file_name=file_name,
            request=request,
        )

    def _construct_request_body(self, prompt: str) -> dict:
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
