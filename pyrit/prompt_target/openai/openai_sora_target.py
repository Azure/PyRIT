# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
from typing import Optional

from openai import APIStatusError, BadRequestError, RateLimitError

from pyrit.exceptions import (
    RateLimitException,
    handle_bad_request_exception,
)
from pyrit.models import (
    Message,
    MessagePiece,
    construct_response_from_request,
    data_serializer_factory,
)
from pyrit.prompt_target import OpenAITarget, limit_requests_per_minute

logger = logging.getLogger(__name__)


class OpenAISoraTarget(OpenAITarget):
    """
    OpenAI Sora Target using the OpenAI SDK for video generation.

    Supports Sora-2 and Sora-2-Pro models via the OpenAI videos API.

    Supported resolutions:
    - Sora-2: 720x1280, 1280x720
    - Sora-2-Pro: 720x1280, 1280x720, 1024x1792, 1792x1024

    Supported durations: 4, 8, or 12 seconds

    Default: model="sora-2", resolution="1280x720", duration=4 seconds
    """

    SUPPORTED_RESOLUTIONS = ["720x1280", "1280x720", "1024x1792", "1792x1024"]
    SUPPORTED_DURATIONS = [4, 8, 12]

    def __init__(
        self,
        *,
        resolution_dimensions: str = "1280x720",
        n_seconds: int = 4,
        **kwargs,
    ):
        """
        Initialize the OpenAI Sora Target.

        Args:
            model_name (str, Optional): The video model to use (e.g., "sora-2", "sora-2-pro").
                Defaults to "sora-2".
            endpoint (str, Optional): The target URL for the OpenAI service.
            api_key (str, Optional): The API key for accessing the service.
                Uses OPENAI_SORA_KEY environment variable by default.
            headers (str, Optional): Extra headers of the endpoint (JSON).
            use_entra_auth (bool, Optional): When set to True, user authentication is used
                instead of API Key.
            max_requests_per_minute (int, Optional): Number of requests the target can handle per
                minute before hitting a rate limit.
            resolution_dimensions (str, Optional): Resolution dimensions for the video in WIDTHxHEIGHT format.
                Defaults to "1280x720".
                Supported resolutions:
                - Sora-2: "720x1280", "1280x720"
                - Sora-2-Pro: "720x1280", "1280x720", "1024x1792", "1792x1024"
            n_seconds (int, Optional): The duration of the generated video (in seconds).
                Defaults to 4. Supported values: 4, 8, or 12 seconds.
        """
        super().__init__(**kwargs)

        # Accept base URLs (/v1), specific API paths (/videos, /v1/videos), Azure formats
        # Note: Only Sora v2 API is supported (uses SDK's videos.create_and_poll)
        sora_url_patterns = [r"/v1$", r"/videos", r"/v1/videos", r"openai/v1", r"\.models\.ai\.azure\.com"]
        self._warn_if_irregular_endpoint(sora_url_patterns)

        self._n_seconds = n_seconds
        self._validate_duration()
        self._size = self._validate_resolution(resolution_dimensions=resolution_dimensions)

    def _set_openai_env_configuration_vars(self) -> None:
        """Set environment variable names."""
        self.model_name_environment_variable = "OPENAI_SORA_MODEL"
        self.endpoint_environment_variable = "OPENAI_SORA_ENDPOINT"
        self.api_key_environment_variable = "OPENAI_SORA_KEY"

    def _validate_resolution(self, *, resolution_dimensions: str) -> str:
        """
        Validate resolution dimensions.

        Args:
            resolution_dimensions: Resolution in WIDTHxHEIGHT format.

        Returns:
            The validated resolution string.

        Raises:
            ValueError: If the resolution is not supported.
        """
        if resolution_dimensions not in self.SUPPORTED_RESOLUTIONS:
            raise ValueError(
                f"Invalid resolution '{resolution_dimensions}'. "
                f"Supported resolutions: {', '.join(self.SUPPORTED_RESOLUTIONS)}"
            )
        return resolution_dimensions

    def _validate_duration(self) -> None:
        """
        Validate video duration.

        Raises:
            ValueError: If the duration is not supported.
        """
        if self._n_seconds not in self.SUPPORTED_DURATIONS:
            raise ValueError(
                f"Invalid duration {self._n_seconds}s. "
                f"Supported durations: {', '.join(map(str, self.SUPPORTED_DURATIONS))} seconds"
            )

    @limit_requests_per_minute
    async def send_prompt_async(self, *, message: Message) -> Message:
        """
        Asynchronously sends a message and generates a video using the OpenAI SDK.

        Args:
            message (Message): The message object containing the prompt.

        Returns:
            Message: The response with the generated video path.

        Raises:
            RateLimitException: If the rate limit is exceeded.
            ValueError: If the request is invalid.
        """
        self._validate_request(message=message)
        request = message.message_pieces[0]
        prompt = request.converted_value

        logger.info(f"Sending video generation prompt: {prompt}")

        try:
            # Use SDK to create video and poll until completion
            video = await self._async_client.videos.create_and_poll(
                model=self._model_name,
                prompt=prompt,
                size=self._size,
                seconds=str(self._n_seconds),
            )

            # Check if video generation was successful
            if video.status == "completed":
                logger.info(f"Video generation completed successfully: {video.id}")
                
                # Download video content using SDK
                video_content = await self._async_client.videos.download_content(video.id)
                
                # Save the video to storage
                return await self._save_video_response(request=request, video_data=video_content)
            
            elif video.status == "failed":
                # Handle failed video generation
                error_message = str(video.error) if video.error else "Video generation failed"
                logger.error(f"Video generation failed: {error_message}")
                
                # Check if it's a content filter error
                is_content_filter = video.error and hasattr(video.error, 'code') and video.error.code in [
                    'content_policy_violation', 
                    'input_moderation',
                    'output_moderation'
                ]
                
                if is_content_filter:
                    return handle_bad_request_exception(
                        response_text=error_message,
                        request=request,
                        is_content_filter=True,
                    )
                else:
                    # Non-content-filter errors are returned as processing errors
                    return construct_response_from_request(
                        request=request,
                        response_text_pieces=[error_message],
                        response_type="error",
                        error="processing",
                    )
            else:
                # Unexpected status
                error_message = f"Video generation ended with unexpected status: {video.status}"
                logger.error(error_message)
                return construct_response_from_request(
                    request=request,
                    response_text_pieces=[error_message],
                    response_type="error",
                    error="unknown",
                )

        except BadRequestError as bre:
            # Handle bad request errors (including content filter)
            error_message = bre.response.text if bre.response else str(bre)
            logger.error(f"Bad request error: {error_message}")
            
            # Check if it's a content filter error
            is_content_filter = False
            try:
                if hasattr(bre, 'body') and bre.body:
                    error_code = bre.body.get('error', {}).get('code', '')
                    is_content_filter = error_code in [
                        'content_policy_violation',
                        'input_moderation',
                        'output_moderation',
                        'content_filter'
                    ]
            except (AttributeError, TypeError):
                pass
            
            return handle_bad_request_exception(
                response_text=error_message,
                request=request,
                error_code=bre.status_code if hasattr(bre, 'status_code') else 400,
                is_content_filter=is_content_filter,
            )
        except RateLimitError:
            logger.error("Rate limit exceeded")
            raise RateLimitException()
        except APIStatusError as e:
            logger.error(f"API error: {e}")
            raise

    async def _save_video_response(self, *, request: MessagePiece, video_data: bytes) -> Message:
        """
        Save video data to storage and construct response.

        Args:
            request: The original request message piece.
            video_data: The video content as bytes.

        Returns:
            Message: The response with the video file path.
        """
        # Save video using data serializer
        data = data_serializer_factory(category="prompt-memory-entries", data_type="video_path")
        await data.save_data(data=video_data)
        video_path = data.value

        logger.info(f"Video saved to: {video_path}")

        # Construct response
        response_entry = construct_response_from_request(
            request=request,
            response_text_pieces=[video_path],
            response_type="video_path",
        )

        return response_entry

    def _validate_request(self, *, message: Message) -> None:
        """
        Validate the request message.

        Args:
            message: The message to validate.

        Raises:
            ValueError: If the request is invalid.
        """
        n_pieces = len(message.message_pieces)
        if n_pieces != 1:
            raise ValueError(f"This target only supports a single message piece. Received: {n_pieces} pieces.")

        piece_type = message.message_pieces[0].converted_value_data_type
        if piece_type != "text":
            raise ValueError(f"This target only supports text prompt input. Received: {piece_type}.")

    def is_json_response_supported(self) -> bool:
        """
        Check if the target supports JSON response data.

        Returns:
            bool: False, as video generation doesn't return JSON content.
        """
        return False
