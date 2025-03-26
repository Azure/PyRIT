# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import json
import logging
from typing import Literal, Optional

import httpx

from pyrit.common import net_utility
from pyrit.exceptions import (
    RateLimitException,
    handle_bad_request_exception,
    pyrit_target_retry,
)
from pyrit.models import (
    PromptRequestResponse,
    construct_response_from_request,
    data_serializer_factory,
)
from pyrit.models.data_type_serializer import VideoPathDataTypeSerializer
from pyrit.prompt_target import OpenAITarget, limit_requests_per_minute

logger = logging.getLogger(__name__)


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
    """

    def __init__(
        self,
        *,
        resolution_dimensions: Literal[
            "360x360", "640x360", "480x480", "854x480", "720x720", "1280x720", "1080x1080", "1920x1080"
        ] = "480x480",
        n_seconds: Optional[int] = 5,
        n_variants: Optional[int] = 1,
        api_version: str = "2025-02-15-preview",
        **kwargs,
    ):
        super().__init__(**kwargs)

        dimensions = resolution_dimensions.split("x")
        self._height = dimensions[1]
        self._width = dimensions[0]

        # Validate input based on resolution dimensions
        if resolution_dimensions in ["1080x1080", "1920x1080"]:
            if n_seconds > 10:
                raise ValueError(
                    "n_seconds must be less than or equal to 10 for resolution dimensions of 1080x1080 or 1920x1080."
                )

            if n_variants > 1:
                raise ValueError(
                    "n_variants must be less than or equal to 1 for resolution dimensions of 1080x1080 or 1920x1080."
                )
        elif resolution_dimensions in ["720x720", "1280x720"]:
            if n_seconds > 20:
                raise ValueError(
                    "n_seconds must be less than or equal to 20 for resolution dimensions other than 1080x1080 or " +
                    "1920x1080."
                )

            if n_variants > 2:
                raise ValueError(
                    "n_variants must be less than or equal to 2 for resolution dimensions of 720x720 or 1280x720."
                )
        else:
            if n_seconds > 20:
                raise ValueError(
                    "n_seconds must be less than or equal to 20 for resolution dimensions other than 1080x1080 or " +
                    "1920x1080."
                )

        self._n_seconds = n_seconds
        self._n_variants = n_variants
        self._api_version = api_version

        self._params = {}
        if self._api_version is not None:
            self._params["api-version"] = self._api_version

        self._in_progress_tasks = set()  # TODO: this should be locked if multiple threads modify

    def _set_openai_env_configuration_vars(self):
        self.model_name_environment_variable = "OPENAI_SORA_MODEL"
        self.endpoint_environment_variable = "OPENAI_SORA_ENDPOINT"
        self.api_key_environment_variable = "OPENAI_SORA_KEY"

    @limit_requests_per_minute
    @pyrit_target_retry
    async def send_prompt_async(self, *, prompt_request: PromptRequestResponse) -> PromptRequestResponse:
        self._validate_request(prompt_request=prompt_request)
        request = prompt_request.request_pieces[0]
        prompt = request.converted_value

        logger.info(f"Sending the following prompt to the prompt target: {prompt}")

        body = self._construct_request_body(prompt=prompt)

        try:
            response = await net_utility.make_request_and_raise_if_error_async(
                endpoint_uri=f"{self._endpoint}/jobs",
                method="POST",
                headers=self._headers,
                request_body=body,
                params=self._params,
                **self._httpx_client_kwargs,
            )
        except httpx.HTTPStatusError as StatusError:
            if StatusError.response.status_code == 400 or (
                StatusError.response.status_code == 500 and "content_filter_results" in StatusError.response.text
            ):
                # Handle Bad Request
                return handle_bad_request_exception(
                    response_text=StatusError.response.text,
                    request=request,
                    error_code=StatusError.response.status_code,
                )
            elif StatusError.response.status_code == 429:
                raise RateLimitException()
            else:
                raise

        response_entry = await self.handle_response(request=request, response=response)

        return response_entry

    # Download video content
    @pyrit_target_retry
    async def download_video_content(self, task_id: str, gen_id: str) -> VideoPathDataTypeSerializer:
        uri = f"{self._endpoint}/{gen_id}/video/content"
        response = await net_utility.make_request_and_raise_if_error_async(
            endpoint_uri=uri,
            method="GET",
            headers=self._headers,
            params=self._params,
            **self._httpx_client_kwargs,
        )

        file_name = f"{task_id}_{gen_id}"
        video_response = data_serializer_factory(
            category="prompt-memory-entries",
            data_type="video_path",
            value=file_name,
        )

        if response.status_code == 200:
            data = response.content
            await video_response.save_data(data=data)

            # TODO: Cleanup by deleting tasks once content downloaded
            # DELETE request for URL f"{endpoint_url}/jobs/{task_id}"
            return video_response
        else:
            raise Exception(f"Failed to download video content: {response}")

    async def handle_response(self, request: PromptRequestResponse, response: httpx.Response) -> PromptRequestResponse:
        # Handle VideoGenerationJob - 201 Created
        content = json.loads(response.content)
        if not content:
            return handle_bad_request_exception(response_text="Empty response", request=request)

        task_id = content.get("id", None)  # Job ID
        status = content.get("status", None)  # Preprocessing, Queued, Processing, Cancelled, Succeeded, Failed
        generations = content.get("generations", [])  # array of VideoGeneration
        failure_reason = content.get("failure_reason", None)  # if status is failed, this will be the reason
        # prompt = content.get('prompt') # the prompt that was used to generate the video

        if status == "succeeded":
            # Download video content
            gen_id = generations[0].get("id", []) if generations else []
            video_response = await self.download_video_content(task_id, gen_id, request)
            response_entry = construct_response_from_request(
                request=request, response_text_pieces=[str(video_response.value)], response_type="video_path"
            )
        elif status != "failed" and status != "cancelled":
            # TODO: Kick off polling in a thread for task status (check_task_status)
            # while status != "succeeded" or "failed" or "cancelled"
            response = await self.check_task_status(task_id=content.get("id", None))
            response_entry = construct_response_from_request(
                request=request,
                response_text_pieces=[str(json.loads(response.content))],
            )
        else:
            if failure_reason == "input_moderation":
                return handle_bad_request_exception(
                    response_text=failure_reason, request=request, is_content_filter=True
                )
            else:
                return handle_bad_request_exception(response_text=failure_reason, request=request)

        return response_entry

    # This is the function that should be called in polling, with the task_id upon
    # first response from the POST call
    @pyrit_target_retry
    async def check_task_status(self, task_id: str) -> httpx.Response:
        """
        Check status of a submitted task using the task_id.
        Args:
            task_id (str): The ID of the task to check.
        Returns:
            httpx.Response: The response from the API.
        """

        uri = f"{self._endpoint}/jobs/{task_id}"

        response = await net_utility.make_request_and_raise_if_error_async(
            endpoint_uri=uri,
            method="GET",
            headers=self._headers,
            params=self._params,
            **self._httpx_client_kwargs,
        )

        return response

    async def check_status_all(self):
        # check status of all in-progress tasks
        for task in self._in_progress_tasks:
            self.check_task_status(task_id=task)

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
