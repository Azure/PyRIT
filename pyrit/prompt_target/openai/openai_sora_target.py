# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import json
import logging
import os
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


# Functions which define when to retry calls to check status and download video
def _retry_check_task(response: httpx.Response) -> bool:
    """
    Returns True if the task status is not "succeeded", "failed", or "cancelled".
    """
    content = json.loads(response.content)
    status = content.get("status", None)  # Preprocessing, Queued, Processing, Cancelled, Succeeded, Failed

    return status not in ["succeeded", "failed", "cancelled"]


def _retry_video_download(response: httpx.Response) -> bool:
    """
    Returns True if the video download status is not 200 (success).
    """
    return response.status_code != 200


class OpenAISoraTarget(OpenAITarget):
    """
    OpenAI Sora Target class for sending prompts to the OpenAI Sora API.

    Args:
        model_name (str, Optional): The name of the model.
        endpoint (str, Optional): The target URL for the OpenAI service.
        api_key (str, Optional): The API key for accessing the Azure OpenAI service.
            Defaults to the OPENAI_CHAT_KEY environment variable.
        headers (str, Optional): Extra headers of the endpoint (JSON).
        use_aad_auth (bool, Optional): When set to True, user authentication is used
            instead of API Key. DefaultAzureCredential is taken for
            https://cognitiveservices.azure.com/.default . Please run `az login` locally
            to leverage user AuthN.
        api_version (str, Optional): The version of the Azure OpenAI API. Defaults to
            "2025-02-15-preview".
        max_requests_per_minute (int, Optional): Number of requests the target can handle per
            minute before hitting a rate limit. The number of requests sent to the target
            will be capped at the value provided.
        httpx_client_kwargs (dict, Optional): Additional kwargs to be passed to the
            httpx.AsyncClient() constructor.
        resolution_dimensions (Literal["360x360", "640x360", "480x480", "854x480", "720x720",
            "1280x720", "1080x1080", "1920x1080"], Optional): Resolution dimensions for the video.
            Defaults to "480x480", where the first value is width and the second is height.
        n_seconds (int, Optional): The number of seconds for the generated video. Defaults to 5.
            For resolutions of 1080p, max seconds is 10. Otherwise, max seconds is 20.
        n_variants (int, Optional): The number of variants for the generated video. Defaults to 1.
            For resolutions of 1080p, max varients is 1. For resolutions of 720p, max varients is 2.
        output_filename (str, Optional): The name of the output file for the generated video.
            Note: DO NOT SET if using target with PromptSendingOrchestrator. The default filename
            is {task_id}_{gen_id}.mp4 as returned by the model.
    """

    # Maximum number of retries for check_task_status()
    # This cannot be set in the constructor as it is used in the decorator, which does not know self.
    try:
        CHECK_TASK_RETRY_MAX_NUM_ATTEMPTS: int = int(os.getenv("CUSTOM_RESULT_RETRY_MAX_NUM_ATTEMPTS", 25))
    except ValueError:
        CHECK_TASK_RETRY_MAX_NUM_ATTEMPTS: int = 25
        logger.warning(
            f"Invalid value for CUSTOM_RESULT_RETRY_MAX_NUM_ATTEMPTS. Using default value of {CHECK_TASK_RETRY_MAX_NUM_ATTEMPTS}."
        )

    def __init__(
        self,
        *,
        resolution_dimensions: Literal[
            "360x360", "640x360", "480x480", "854x480", "720x720", "1280x720", "1080x1080", "1920x1080"
        ] = "480x480",
        n_seconds: Optional[int] = 5,
        n_variants: Optional[int] = 1,
        api_version: str = "2025-02-15-preview",
        output_filename: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        dimensions = resolution_dimensions.split("x")
        self._height = dimensions[1]
        self._width = dimensions[0]

        # Validate input based on resolution dimensions
        self._validate_video_constraints(
            resolution_dimensions=resolution_dimensions,
            n_seconds=n_seconds,
            n_variants=n_variants,
        )

        self._n_seconds = n_seconds
        self._n_variants = n_variants
        self._api_version = api_version
        self._output_filename = output_filename

        self._params = {}
        if self._api_version is not None:
            self._params["api-version"] = self._api_version

    def _set_openai_env_configuration_vars(self):
        self.model_name_environment_variable = "OPENAI_SORA_MODEL"
        self.endpoint_environment_variable = "OPENAI_SORA_ENDPOINT"
        self.api_key_environment_variable = "OPENAI_SORA_KEY"

    async def _send_httpx_request(
        self,
        *,
        endpoint_uri: str,
        method: str,
        request_body: dict[str, object] = None
    ) -> httpx.Response:
        """
        Send an HTTP request using the httpx client and handle exceptions.

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
        self._validate_request(prompt_request=prompt_request)
        request = prompt_request.request_pieces[0]
        prompt = request.converted_value

        logger.info(f"Sending the following prompt to the prompt target: {prompt}")

        body = self._construct_request_body(prompt=prompt)

        response = await self._send_httpx_request(
            endpoint_uri=f"{self._endpoint}/jobs",
            method="POST",
            request_body=body,
        )

        return await self._handle_response(request=request, response=response)

    @pyrit_custom_result_retry(
        retry_function=_retry_check_task,
        retry_max_num_attempts=CHECK_TASK_RETRY_MAX_NUM_ATTEMPTS,
    )
    @pyrit_target_retry
    async def check_task_status(self, task_id: str) -> httpx.Response:
        f"""
        Check status of a submitted task using the task_id.
        Retries a maxium of {self.CHECK_TASK_RETRY_MAX_NUM_ATTEMPTS} times, 
        until the task is complete (succeeded, failed, or cancelled). Also
        retries upon RateLimitException.

        Args:
            task_id (str): The ID of the task to check.
        Returns:
            httpx.Response: The response from the API.
        Raises:
            RateLimitException: If the rate limit is exceeded.
            httpx.HTTPStatusError: If the request fails.
        """

        uri = f"{self._endpoint}/jobs/{task_id}"

        response = await self._send_httpx_request(
            endpoint_uri=uri,
            method="GET",
        )

        return response

    @pyrit_custom_result_retry(
        retry_function=_retry_video_download,
    )
    @pyrit_target_retry
    async def download_video_content(self, gen_id: str) -> httpx.Response:
        """
        Download the video using the generation ID.
        Retries if the response status code is not 200. Also retries upon RateLimitException.

        Args:
            gen_id (str): The ID of the generation to check.
        Returns:
            httpx.Response: The response from the API.
        Raises:
            RateLimitException: If the rate limit is exceeded.
            httpx.HTTPStatusError: If the request fails.
        """

        logger.info(f"Downloading video content for generation ID: {gen_id}")
        uri = f"{self._endpoint}/{gen_id}/video/content"

        response = await self._send_httpx_request(
            endpoint_uri=uri,
            method="GET",
        )

        return response

    async def _save_video_to_storage(
        self,
        *,
        data: bytes,
        task_id: str,
        gen_id: str,
        request: PromptRequestPiece,
    ) -> PromptRequestResponse:
        serializer = data_serializer_factory(
            category="prompt-memory-entries",
            data_type="video_path",
        )
        file_name = self._output_filename if self._output_filename else f"{task_id}_{gen_id}"
        await serializer.save_data(data=data, output_filename=file_name)
        logger.info(f"Video response saved successfully to {serializer.value}")

        response_entry = construct_response_from_request(
            request=request, response_text_pieces=[str(serializer.value)], response_type="video_path"
        )

        return response_entry

    async def _handle_response(self, request: PromptRequestPiece, response: httpx.Response) -> PromptRequestResponse:
        """
        Handle the response to a video generation request.
        This includes checking the status of the task and downloading the video content if successful.
        """
        content = json.loads(response.content)

        task_id = content.get("id")
        logger.info(f"Handling response for Task ID: {task_id}")

        # Check status with retry until task is complete (succeeded, failed, or cancelled)
        task_response = await self.check_task_status(task_id=task_id)
        task_content = json.loads(task_response.content)
        status = task_content.get("status")

        # Handle completed task
        if status == "succeeded":
            # Download video content
            generations = task_content.get("generations")
            gen_id = generations[0].get("id")

            video_response = await self.download_video_content(gen_id=gen_id)
            response_entry = await self._save_video_to_storage(
                data=video_response.content,
                task_id=task_id,
                gen_id=gen_id,
                request=request,
            )
        elif status in ["failed", "cancelled"]:
            # Handle failed or cancelled task
            failure_reason = task_content.get("failure_reason", None)
            failure_message = f"{task_id} {status}, Reason: {failure_reason}"
            logger.error(failure_message)

            if failure_reason in ["input_moderation", "output_moderation"]:
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
                f"{task_id} is still processing after attempting retries. Consider setting a value > {self.CHECK_TASK_RETRY_MAX_NUM_ATTEMPTS} "
                + f"for environment variable CUSTOM_RESULT_RETRY_MAX_NUM_ATTEMPTS. Status: {status}"
            )
            response_entry = construct_response_from_request(
                request=request,
                response_text_pieces=[f"{str(task_content)}"],
            )

        return response_entry

    def _validate_video_constraints(
            self,
            resolution_dimensions: str,
            n_variants: int,
            n_seconds: int
        ) -> None:
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

    def _construct_request_body(self, prompt: str) -> dict:

        body_parameters: dict[str, object] = {
            "prompt": prompt,
            "height": self._height,
            "width": self._width,
            "n_seconds": self._n_seconds,
            "n_variants": self._n_variants,
        }

        # Filter out None values
        return {k: v for k, v in body_parameters.items() if v is not None}

    def _validate_request(self, *, prompt_request: PromptRequestResponse) -> None:
        if len(prompt_request.request_pieces) != 1:
            raise ValueError("This target only supports a single prompt request piece.")

        if prompt_request.request_pieces[0].converted_value_data_type != "text":
            raise ValueError("This target only supports text prompt input.")

        request = prompt_request.request_pieces[0]
        messages = self._memory.get_chat_messages_with_conversation_id(conversation_id=request.conversation_id)

        if len(messages) > 0:
            raise ValueError("This target only supports a single turn conversation.")

    def is_json_response_supported(self) -> bool:
        """Indicates that this target supports JSON response format."""
        return False
