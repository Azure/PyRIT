# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
import os
from typing import Any, Optional

from pyrit.exceptions import (
    pyrit_target_retry,
)
from pyrit.models import (
    DataTypeSerializer,
    Message,
    MessagePiece,
    construct_response_from_request,
    data_serializer_factory,
)
from pyrit.prompt_target.common.utils import limit_requests_per_minute
from pyrit.prompt_target.openai.openai_error_handling import _is_content_filter_error
from pyrit.prompt_target.openai.openai_target import OpenAITarget

logger = logging.getLogger(__name__)


class OpenAIVideoTarget(OpenAITarget):
    """
    OpenAI Video Target using the OpenAI SDK for video generation.

    Supports Sora-2 and Sora-2-Pro models via the OpenAI videos API.

    Supports three modes:
    - Text-to-video: Generate video from a text prompt
    - Image-to-video: Generate video using an image as the first frame (include image_path piece)
    - Remix: Create variation of existing video (include video_id in prompt_metadata)

    Supported resolutions:
    - Sora-2: 720x1280, 1280x720
    - Sora-2-Pro: 720x1280, 1280x720, 1024x1792, 1792x1024

    Supported durations: 4, 8, or 12 seconds

    Default: resolution="1280x720", duration=4 seconds

    Supported image formats for image-to-video: JPEG, PNG, WEBP
    """

    SUPPORTED_RESOLUTIONS = ["720x1280", "1280x720", "1024x1792", "1792x1024"]
    SUPPORTED_DURATIONS = [4, 8, 12]

    def __init__(
        self,
        *,
        resolution_dimensions: str = "1280x720",
        n_seconds: int = 4,
        **kwargs: Any,
    ) -> None:
        """
        Initialize the OpenAI Video Target.

        Args:
            model_name (str, Optional): The video model to use (e.g., "sora-2", "sora-2-pro")
                (or deployment name in Azure). If no value is provided, the OPENAI_VIDEO_MODEL
                environment variable will be used.
            endpoint (str, Optional): The target URL for the OpenAI service.
            api_key (str | Callable[[], str], Optional): The API key for accessing the OpenAI service,
                or a callable that returns an access token. For Azure endpoints with Entra authentication,
                pass a token provider from pyrit.auth (e.g., get_azure_openai_auth(endpoint)).
                Uses OPENAI_VIDEO_KEY environment variable by default.
            headers (str, Optional): Extra headers of the endpoint (JSON).
            max_requests_per_minute (int, Optional): Number of requests the target can handle per
                minute before hitting a rate limit.
            resolution_dimensions (str, Optional): Resolution dimensions for the video in WIDTHxHEIGHT format.
                Defaults to "1280x720".
                Supported resolutions:
                - Sora-2: "720x1280", "1280x720"
                - Sora-2-Pro: "720x1280", "1280x720", "1024x1792", "1792x1024"
            n_seconds (int, Optional): The duration of the generated video (in seconds).
                Defaults to 4. Supported values: 4, 8, or 12 seconds.
            **kwargs: Additional keyword arguments passed to the parent OpenAITarget class.
            httpx_client_kwargs (dict, Optional): Additional kwargs to be passed to the ``httpx.AsyncClient()``
                constructor. For example, to specify a 3 minute timeout: ``httpx_client_kwargs={"timeout": 180}``
        """
        super().__init__(**kwargs)

        self._n_seconds = n_seconds
        self._validate_duration()
        self._size = self._validate_resolution(resolution_dimensions=resolution_dimensions)

    def _set_openai_env_configuration_vars(self) -> None:
        """Set environment variable names."""
        self.model_name_environment_variable = "OPENAI_VIDEO_MODEL"
        self.endpoint_environment_variable = "OPENAI_VIDEO_ENDPOINT"
        self.api_key_environment_variable = "OPENAI_VIDEO_KEY"
        self.underlying_model_environment_variable = "OPENAI_VIDEO_UNDERLYING_MODEL"

    def _get_target_api_paths(self) -> list[str]:
        """Return API paths that should not be in the URL."""
        return ["/videos", "/v1/videos"]

    def _get_provider_examples(self) -> dict[str, str]:
        """Return provider-specific example URLs."""
        return {
            ".openai.azure.com": "https://{resource}.openai.azure.com/openai/v1",
            "api.openai.com": "https://api.openai.com/v1",
        }

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
    @pyrit_target_retry
    async def send_prompt_async(self, *, message: Message) -> list[Message]:
        """
        Asynchronously sends a message and generates a video using the OpenAI SDK.

        Supports three modes:
        - Text-to-video: Single text piece
        - Image-to-video: Text piece + image_path piece (image becomes first frame)
        - Remix: Text piece with prompt_metadata["video_id"] set to an existing video ID

        Args:
            message (Message): The message object containing the prompt.

        Returns:
            list[Message]: A list containing the response with the generated video path.

        Raises:
            RateLimitException: If the rate limit is exceeded.
            ValueError: If the request is invalid.
        """
        self._validate_request(message=message)

        # Extract pieces by type
        pieces = message.message_pieces
        text_piece = next(p for p in pieces if p.converted_value_data_type == "text")
        image_piece = next((p for p in pieces if p.converted_value_data_type == "image_path"), None)
        prompt = text_piece.converted_value

        # Check for remix mode via prompt_metadata
        remix_video_id = text_piece.prompt_metadata.get("video_id") if text_piece.prompt_metadata else None

        logger.info(f"Sending video generation prompt: {prompt}")

        if remix_video_id:
            # REMIX MODE: Create variation of existing video
            logger.info(f"Remix mode: Creating variation of video {remix_video_id}")
            response = await self._handle_openai_request(
                api_call=lambda: self._remix_and_poll_async(video_id=remix_video_id, prompt=prompt),
                request=message,
            )
        elif image_piece:
            # IMAGE-TO-VIDEO MODE: Use image as first frame
            logger.info("Image-to-video mode: Using image as first frame")
            image_path = image_piece.converted_value
            image_serializer = data_serializer_factory(
                value=image_path, data_type="image_path", category="prompt-memory-entries"
            )
            image_bytes = await image_serializer.read_data()

            # Get MIME type for proper file upload (API requires content-type)
            mime_type = DataTypeSerializer.get_mime_type(image_path)
            if not mime_type:
                # Default to PNG if MIME type cannot be determined
                mime_type = "image/png"

            # Create file tuple with filename and MIME type for OpenAI SDK
            # Format: (filename, content, content_type)
            filename = os.path.basename(image_path)
            input_file = (filename, image_bytes, mime_type)

            response = await self._handle_openai_request(
                api_call=lambda: self._async_client.videos.create_and_poll(
                    model=self._model_name,
                    prompt=prompt,
                    size=self._size,  # type: ignore[arg-type]
                    seconds=str(self._n_seconds),  # type: ignore[arg-type]
                    input_reference=input_file,
                ),
                request=message,
            )
        else:
            # TEXT-TO-VIDEO MODE: Standard generation
            response = await self._handle_openai_request(
                api_call=lambda: self._async_client.videos.create_and_poll(
                    model=self._model_name,
                    prompt=prompt,
                    size=self._size,  # type: ignore[arg-type]
                    seconds=str(self._n_seconds),  # type: ignore[arg-type]
                ),
                request=message,
            )

        return [response]

    async def _remix_and_poll_async(self, *, video_id: str, prompt: str) -> Any:
        """
        Create a remix of an existing video and poll until complete.

        The OpenAI SDK's remix() method returns immediately with a job status.
        This method polls until the job completes or fails.

        Args:
            video_id: The ID of the completed video to remix.
            prompt: The text prompt directing the remix.

        Returns:
            The completed Video object from the OpenAI SDK.
        """
        video = await self._async_client.videos.remix(video_id, prompt=prompt)

        # Poll until completion if not already done
        if video.status not in ["completed", "failed"]:
            video = await self._async_client.videos.poll(video.id)

        return video

    def _check_content_filter(self, response: Any) -> bool:
        """
        Check if a video generation response was content filtered.

        Response indicates content filtering through:
        - Status is "failed"
        - Error code is "content_filter" (output-side filtering)
        - Error code is "moderation_blocked" (input moderation)

        Note: Input-side filtering (content_policy_violation via BadRequestError) is also caught
        by the base class before reaching this method.

        Args:
            response: A Video object from the OpenAI SDK.

        Returns:
            True if content was filtered, False otherwise.
        """
        if response.status == "failed" and response.error:
            # Convert response to dict and use common filter detection
            response_dict = response.model_dump()
            return _is_content_filter_error(response_dict)
        return False

    async def _construct_message_from_response(self, response: Any, request: Any) -> Message:
        """
        Construct a Message from a video response.

        Args:
            response: The Video response from OpenAI SDK.
            request: The original request MessagePiece.

        Returns:
            Message: Constructed message with video file path.
        """
        video = response

        # Check if video generation was successful
        if video.status == "completed":
            logger.info(f"Video generation completed successfully: {video.id}")

            # Log remix metadata if available
            if hasattr(video, "remixed_from_video_id") and video.remixed_from_video_id:
                logger.info(f"Video was remixed from: {video.remixed_from_video_id}")

            # Download video content using SDK
            video_response = await self._async_client.videos.download_content(video.id)
            # Extract bytes from HttpxBinaryResponseContent
            video_content = video_response.content

            # Save the video to storage (include video.id for chaining remixes)
            return await self._save_video_response(request=request, video_data=video_content, video_id=video.id)

        elif video.status == "failed":
            # Handle failed video generation (non-content-filter)
            error_message = str(video.error) if video.error else "Video generation failed"
            logger.error(f"Video generation failed: {error_message}")

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

    async def _save_video_response(
        self, *, request: MessagePiece, video_data: bytes, video_id: Optional[str] = None
    ) -> Message:
        """
        Save video data to storage and construct response.

        Args:
            request: The original request message piece.
            video_data: The video content as bytes.
            video_id: The video ID from the API (stored in metadata for chaining remixes).

        Returns:
            Message: The response with the video file path.
        """
        # Save video using data serializer
        data = data_serializer_factory(category="prompt-memory-entries", data_type="video_path")
        await data.save_data(data=video_data)
        video_path = data.value

        logger.info(f"Video saved to: {video_path}")

        # Include video_id in metadata for chaining (e.g., remix the generated video later)
        prompt_metadata = {"video_id": video_id} if video_id else None

        # Construct response
        response_entry = construct_response_from_request(
            request=request,
            response_text_pieces=[video_path],
            response_type="video_path",
            prompt_metadata=prompt_metadata,
        )

        return response_entry

    def _validate_request(self, *, message: Message) -> None:
        """
        Validate the request message.

        Accepts:
        - Single text piece (text-to-video or remix mode)
        - Text piece + image_path piece (image-to-video mode)

        Args:
            message: The message to validate.

        Raises:
            ValueError: If the request is invalid.
        """
        pieces = message.message_pieces
        n_pieces = len(pieces)

        if n_pieces == 0:
            raise ValueError("Message must contain at least one piece.")

        # Categorize pieces
        text_pieces = [p for p in pieces if p.converted_value_data_type == "text"]
        image_pieces = [p for p in pieces if p.converted_value_data_type == "image_path"]
        other_pieces = [p for p in pieces if p.converted_value_data_type not in ("text", "image_path")]

        # Must have exactly one text piece
        if len(text_pieces) != 1:
            raise ValueError(f"Expected exactly 1 text piece, got {len(text_pieces)}.")

        # At most one image piece
        if len(image_pieces) > 1:
            raise ValueError(f"Expected at most 1 image piece, got {len(image_pieces)}.")

        # No other data types allowed
        if other_pieces:
            types = [p.converted_value_data_type for p in other_pieces]
            raise ValueError(f"Unsupported piece types: {types}. Only 'text' and 'image_path' are supported.")

        # Check for conflicting modes: remix + image
        text_piece = text_pieces[0]
        remix_video_id = text_piece.prompt_metadata.get("video_id") if text_piece.prompt_metadata else None
        if remix_video_id and image_pieces:
            raise ValueError("Cannot use image input in remix mode. Remix uses existing video as reference.")

    def is_json_response_supported(self) -> bool:
        """
        Check if the target supports JSON response data.

        Returns:
            bool: False, as video generation doesn't return JSON content.
        """
        return False
