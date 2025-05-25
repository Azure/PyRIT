# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import json
import logging
from typing import Any, MutableSequence, Optional

import httpx

from pyrit.common import net_utility
from pyrit.exceptions import (
    EmptyResponseException,
    PyritException,
    handle_bad_request_exception,
    pyrit_target_retry,
)
from pyrit.exceptions.exception_classes import RateLimitException
from pyrit.models import (
    ChatMessageListDictContent,
    DataTypeSerializer,
    PromptRequestPiece,
    PromptRequestResponse,
    data_serializer_factory,
)
from pyrit.models.literals import PromptDataType
from pyrit.prompt_target import OpenAITarget, limit_requests_per_minute

logger = logging.getLogger(__name__)

# Supported image formats for Azure OpenAI GPT-4o,
# https://learn.microsoft.com/en-us/azure/ai-services/openai/concepts/use-your-image-data
AZURE_OPENAI_GPT4O_SUPPORTED_IMAGE_FORMATS = [".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", "tif"]


class OpenAIResponseTarget(OpenAITarget):
    """
    This class facilitates multimodal (image and text) input and text output generation

    This works with GPT3.5, GPT4, GPT4o, GPT-V, and other compatible models
    """

    def __init__(
        self,
        *,
        api_version: Optional[str] = "2025-04-16",
        max_output_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        extra_body_parameters: Optional[dict[str, Any]] = None,
        **kwargs,
    ):
        """
        Args:
            model_name (str, Optional): The name of the model.
            endpoint (str, Optional): The target URL for the OpenAI service.
            api_key (str, Optional): The API key for accessing the Azure OpenAI service.
                Defaults to the OPENAI_RESPONSES_KEY environment variable.
            headers (str, Optional): Headers of the endpoint (JSON).
            max_requests_per_minute (int, Optional): Number of requests the target can handle per
                minute before hitting a rate limit. The number of requests sent to the target
                will be capped at the value provided.
            max_output_tokens (int, Optional): The maximum number of tokens that can be
                generated in the response. This value can be used to control
                costs for text generated via API.
            temperature (float, Optional): The temperature parameter for controlling the
                randomness of the response.
            top_p (float, Optional): The top-p parameter for controlling the diversity of the
                response.
            extra_body_parameters (dict, Optional): Additional parameters to be included in the request body.
        """
        super().__init__(api_version=api_version, **kwargs)

        self._max_output_tokens = max_output_tokens
        self._temperature = temperature
        self._top_p = top_p
        self._extra_body_parameters = extra_body_parameters

    def _set_openai_env_configuration_vars(self) -> None:
        self.model_name_environment_variable = "OPENAI_RESPONSES_MODEL"
        self.endpoint_environment_variable = "OPENAI_RESPONSES_ENDPOINT"
        self.api_key_environment_variable = "OPENAI_RESPONSES_KEY"

    @limit_requests_per_minute
    @pyrit_target_retry
    async def send_prompt_async(self, *, prompt_request: PromptRequestResponse) -> PromptRequestResponse:
        """Asynchronously sends a prompt request and handles the response within a managed conversation context.

        Args:
            prompt_request (PromptRequestResponse): The prompt request response object.

        Returns:
            PromptRequestResponse: The updated conversation entry with the response from the prompt target.
        """

        self._validate_request(prompt_request=prompt_request)
        self.refresh_auth_headers()

        request_piece: PromptRequestPiece = prompt_request.request_pieces[0]

        is_json_response = self.is_response_format_json(request_piece)

        conversation = self._memory.get_conversation(conversation_id=request_piece.conversation_id)
        conversation.append(prompt_request)

        logger.info(f"Sending the following prompt to the prompt target: {prompt_request}")

        body = await self._construct_request_body(conversation=conversation, is_json_response=is_json_response)

        params = {}
        if self._api_version is not None:
            params["api-version"] = self._api_version

        try:
            str_response: httpx.Response = await net_utility.make_request_and_raise_if_error_async(
                endpoint_uri=self._endpoint,
                method="POST",
                headers=self._headers,
                request_body=body,
                params=params,
            )
        except httpx.HTTPStatusError as StatusError:
            if StatusError.response.status_code == 400:
                # Handle Bad Request
                return handle_bad_request_exception(
                    response_text=StatusError.response.text,
                    request=request_piece,
                    error_code=StatusError.response.status_code,
                )
            elif StatusError.response.status_code == 429:
                raise RateLimitException()
            else:
                raise

        logger.info(f'Received the following response from the prompt target "{str_response.text}"')
        response: PromptRequestResponse = self._construct_prompt_response_from_openai_json(
            open_ai_str_response=str_response.text, request_piece=request_piece
        )

        return response

    async def _convert_local_image_to_data_url(self, image_path: str) -> str:
        """Converts a local image file to a data URL encoded in base64.

        Args:
            image_path (str): The file system path to the image file.

        Raises:
            FileNotFoundError: If no file is found at the specified `image_path`.
            ValueError: If the image file's extension is not in the supported formats list.

        Returns:
            str: A string containing the MIME type and the base64-encoded data of the image, formatted as a data URL.
        """
        ext = DataTypeSerializer.get_extension(image_path)
        if ext.lower() not in AZURE_OPENAI_GPT4O_SUPPORTED_IMAGE_FORMATS:
            raise ValueError(
                f"Unsupported image format: {ext}. Supported formats are: {AZURE_OPENAI_GPT4O_SUPPORTED_IMAGE_FORMATS}"
            )

        mime_type = DataTypeSerializer.get_mime_type(image_path)
        if not mime_type:
            mime_type = "application/octet-stream"

        image_serializer = data_serializer_factory(
            category="prompt-memory-entries", value=image_path, data_type="image_path", extension=ext
        )
        base64_encoded_data = await image_serializer.read_data_base64()
        # Azure OpenAI GPT-4o documentation doesn't specify the local image upload format for API.
        # GPT-4o image upload format is determined using "view code" functionality in Azure OpenAI deployments
        # The image upload format is same as GPT-4 Turbo.
        # Construct the data URL, as per Azure OpenAI GPT-4 Turbo local image format
        # https://learn.microsoft.com/en-us/azure/ai-services/openai/how-to/gpt-with-vision?tabs=rest%2Csystem-assigned%2Cresource#call-the-chat-completion-apis
        return f"data:{mime_type};base64,{base64_encoded_data}"

    async def _build_input_for_multi_modal_async(
        self, conversation: MutableSequence[PromptRequestResponse]
    ) -> list[dict]:
        """
        Builds chat messages based on prompt request response entries.

        Args:
            conversation (list[PromptRequestResponse]): A list of PromptRequestResponse objects.

        Returns:
            list[dict]: The list of constructed chat messages.
        """
        full_request = []
        for message in conversation:

            prompt_request_pieces = message.request_pieces

            if len(prompt_request_pieces) == 0:
                raise ValueError("No prompt request pieces found in the conversation.")

            content = []
            for prompt_request_piece in prompt_request_pieces:
                role = prompt_request_piece.role
                if prompt_request_piece.converted_value_data_type == "text":
                    if role == "user":
                        type = "input_text"
                    else:
                        type = "output_text"
                    entry = {"type": type, "text": prompt_request_piece.converted_value}
                    content.append(entry)
                elif prompt_request_piece.converted_value_data_type == "image_path":
                    data_base64_encoded_url = await self._convert_local_image_to_data_url(
                        prompt_request_piece.converted_value
                    )
                    image_url_entry = {"url": data_base64_encoded_url}
                    entry = {"type": "input_image", "image_url": image_url_entry}  # type: ignore
                    content.append(entry)
                else:
                    raise ValueError(
                        f"Multimodal data type {prompt_request_piece.converted_value_data_type} is not yet supported."
                    )

            full_request.append(
                ChatMessageListDictContent(role=role, content=content).model_dump(exclude_none=True)  # type: ignore
            )

        return full_request

    async def _construct_request_body(
        self, conversation: MutableSequence[PromptRequestResponse], is_json_response: bool
    ) -> dict:
        input = await self._build_input_for_multi_modal_async(conversation)

        body_parameters = {
            "model": self._model_name,
            "max_output_tokens": self._max_output_tokens,
            "temperature": self._temperature,
            "top_p": self._top_p,
            "stream": False,
            "input": input,
            "response_format": {"type": "json_object"} if is_json_response else None,
        }

        if self._extra_body_parameters:
            for key, value in self._extra_body_parameters.items():
                body_parameters[key] = value

        # Filter out None values
        return {k: v for k, v in body_parameters.items() if v is not None}

    def _construct_prompt_response_from_openai_json(
        self,
        *,
        open_ai_str_response: str,
        request_piece: PromptRequestPiece,
    ) -> PromptRequestResponse:

        try:
            response = json.loads(open_ai_str_response)
        except json.JSONDecodeError as e:
            raise PyritException(message=f"Failed to parse JSON response. Please check your endpoint: {e}")

        status = response.get("status")
        error = response.get("error")
        extracted_response_pieces = []
        if status is None:
            if error and "code" in error and error["code"] == "content_filter":
                # TODO validate that this is correct with AOAI
                return handle_bad_request_exception(
                    response_text=open_ai_str_response, request=request_piece, error_code=400, is_content_filter=True
                )
            else:
                raise PyritException(message=f"Unexpected response format: {response}. Expected 'status' key.")
        elif status != "completed" or error is not None:
            raise PyritException(message=f"Status {status} and error {error} from response: {response}")
        else:
            for piece in response["output"]:
                piece_type: PromptDataType = "text"
                if piece["type"] == "message":
                    piece_value = piece["content"][0]["text"]
                elif piece["type"] == "reasoning":
                    # TODO: add option to request reasoning summaries
                    # TODO: ensure proper serialization of reasoning summaries
                    piece_value = ""
                    piece_type = "reasoning"
                    for summary_piece in piece["content"][0]["summary"]:
                        if summary_piece["type"] == "summary_text":
                            piece_value += summary_piece["text"]
                    if not piece_value:
                        continue  # Skip empty reasoning summaries
                else:
                    # other options include "image_generation_call", "file_search_call", "function_call",
                    # "web_search_call", "computer_call", "code_interpreter_call", "local_shell_call",
                    # "mcp_call", "mcp_list_tools", "mcp_approval_request"
                    raise ValueError(
                        f"Unsupported response type: {piece['type']}. PyRIT does not yet support this response type."
                    )

                # Handle empty response
                if not piece_value:
                    logger.log(logging.ERROR, "The chat returned an empty response.")
                    raise EmptyResponseException(message="The chat returned an empty response.")

                extracted_response_pieces.append(
                    PromptRequestPiece(
                        role="assistant",
                        original_value=piece_value,
                        conversation_id=request_piece.conversation_id,
                        labels=request_piece.labels,
                        prompt_target_identifier=request_piece.prompt_target_identifier,
                        orchestrator_identifier=request_piece.orchestrator_identifier,
                        original_value_data_type=piece_type,
                        response_error=error or "none",
                    )
                )

        if not extracted_response_pieces:
            raise PyritException(message="No valid response pieces found in the response.")

        return PromptRequestResponse(request_pieces=extracted_response_pieces)

    def _validate_request(self, *, prompt_request: PromptRequestResponse) -> None:
        """Validates the structure and content of a prompt request for compatibility of this target.

        Args:
            prompt_request (PromptRequestResponse): The prompt request response object.

        Raises:
            ValueError: If more than two request pieces are provided.
            ValueError: If any of the request pieces have a data type other than 'text' or 'image_path'.
        """

        converted_prompt_data_types = [
            request_piece.converted_value_data_type for request_piece in prompt_request.request_pieces
        ]

        # Some models may not support all of these
        for prompt_data_type in converted_prompt_data_types:
            if prompt_data_type not in ["text", "image_path"]:
                raise ValueError("This target only supports text and image_path.")

    def is_json_response_supported(self) -> bool:
        """Indicates that this target supports JSON response format."""
        return True
