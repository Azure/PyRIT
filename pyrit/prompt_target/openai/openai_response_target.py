# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import json
import logging
from enum import Enum
from typing import Any, Dict, List, MutableSequence, Optional, Sequence

from pyrit.common import convert_local_image_to_data_url
from pyrit.exceptions import (
    EmptyResponseException,
    PyritException,
    handle_bad_request_exception,
)
from pyrit.models import (
    ChatMessage,
    ChatMessageListDictContent,
    ChatMessageRole,
    PromptDataType,
    PromptRequestPiece,
    PromptRequestResponse,
    PromptResponseError,
)
from pyrit.prompt_target.openai.openai_chat_target_base import OpenAIChatTargetBase

logger = logging.getLogger(__name__)


# Define PromptRequestPieceType enum for all mentioned types


class PromptRequestPieceType(str, Enum):
    MESSAGE = "message"
    REASONING = "reasoning"
    IMAGE_GENERATION_CALL = "image_generation_call"
    FILE_SEARCH_CALL = "file_search_call"
    FUNCTION_CALL = "function_call"
    WEB_SEARCH_CALL = "web_search_call"
    COMPUTER_CALL = "computer_call"
    CODE_INTERPRETER_CALL = "code_interpreter_call"
    LOCAL_SHELL_CALL = "local_shell_call"
    MCP_CALL = "mcp_call"
    MCP_LIST_TOOLS = "mcp_list_tools"
    MCP_APPROVAL_REQUEST = "mcp_approval_request"


class OpenAIResponseTarget(OpenAIChatTargetBase):
    """
    This class enables communication with endpoints that support the OpenAI Response API.

    This works with models such as o1, o3, and o4-mini.
    Depending on the endpoint this allows for a variety of inputs, outputs, and tool calls.
    For more information, see the OpenAI Response API documentation:
    https://platform.openai.com/docs/api-reference/responses/create
    """

    def __init__(
        self,
        *,
        api_version: Optional[str] = "2025-03-01-preview",
        max_output_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        extra_body_parameters: Optional[dict[str, Any]] = None,
        **kwargs,
    ):
        """
        Initializes the OpenAIResponseTarget with the provided parameters.

        Args:
            model_name (str, Optional): The name of the model.
            endpoint (str, Optional): The target URL for the OpenAI service.
            api_key (str, Optional): The API key for accessing the Azure OpenAI service.
                Defaults to the OPENAI_RESPONSES_KEY environment variable.
            headers (str, Optional): Headers of the endpoint (JSON).
            api_version (str, Optional): The version of the Azure OpenAI API. Defaults to
                "2025-03-01-preview".
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
            is_json_supported (bool, Optional): If True, the target will support formatting responses as JSON by
                setting the response_format header. Official OpenAI models all support this, but if you are using
                this target with different models, is_json_supported should be set correctly to avoid issues when
                using adversarial infrastructure (e.g. Crescendo scorers will set this flag).
            extra_body_parameters (dict, Optional): Additional parameters to be included in the request body.
            httpx_client_kwargs (dict, Optional): Additional kwargs to be passed to the
                httpx.AsyncClient() constructor.
                For example, to specify a 3 minute timeout: httpx_client_kwargs={"timeout": 180}

        Raises:
            PyritException: If the temperature or top_p values are out of bounds.
            ValueError: If the temperature is not between 0 and 2 (inclusive).
            ValueError: If the top_p is not between 0 and 1 (inclusive).
            ValueError: If both `max_output_tokens` and `max_tokens` are provided.
            RateLimitException: If the target is rate-limited.
            httpx.HTTPStatusError: If the request fails with a 400 Bad Request or 429 Too Many Requests error.
            json.JSONDecodeError: If the response from the target is not valid JSON.
            Exception: If the request fails for any other reason.
        """
        super().__init__(api_version=api_version, temperature=temperature, top_p=top_p, **kwargs)

        self._max_output_tokens = max_output_tokens

        # Reasoning parameters are not yet supported by PyRIT.
        # See https://platform.openai.com/docs/api-reference/responses/create#responses-create-reasoning
        # for more information.

        self._extra_body_parameters = extra_body_parameters

    def _set_openai_env_configuration_vars(self) -> None:
        self.model_name_environment_variable = "OPENAI_RESPONSES_MODEL"
        self.endpoint_environment_variable = "OPENAI_RESPONSES_ENDPOINT"
        self.api_key_environment_variable = "OPENAI_RESPONSES_KEY"

    async def _build_input_for_multi_modal_async(
        self, conversation: MutableSequence[PromptRequestResponse]
    ) -> List[Dict[str, Any]]:
        """
        Builds chat messages based on prompt request response entries.

        Args:
            conversation: A list of PromptRequestResponse objects representing the conversation.

        Returns:
            A list of constructed chat messages formatted for OpenAI Response API.

        Raises:
            ValueError: If conversation is empty or contains invalid message structure.
        """
        if not conversation:
            raise ValueError("Conversation cannot be empty")

        chat_messages: List[Dict[str, Any]] = []

        for message_index, message in enumerate(conversation):
            try:
                chat_message = await self._process_single_message(message)
                if chat_message:  # Only add non-empty messages
                    chat_messages.append(chat_message)
            except Exception as e:
                logger.error(f"Failed to process message at index {message_index}: {e}")
                raise ValueError(f"Failed to process conversation message at index {message_index}: {e}") from e

        self._translate_roles(conversation=chat_messages)
        return chat_messages

    async def _process_single_message(self, message: PromptRequestResponse) -> Optional[Dict[str, Any]]:
        """
        Process a single message from the conversation.

        Args:
            message: The PromptRequestResponse to process.

        Returns:
            A formatted chat message dictionary, or None if message should be skipped.

        Raises:
            ValueError: If message structure is invalid.
        """
        request_pieces = message.request_pieces

        if not request_pieces:
            raise ValueError("Message contains no request pieces")

        # Handle system messages as a special case
        first_piece = request_pieces[0]
        if first_piece.role == "system":
            return self._create_system_message(request_pieces)

        # Process regular messages with content
        content_items = await self._build_content_items(request_pieces)

        # Skip messages with no valid content
        if not content_items:
            logger.warning("Skipping message with no valid content items")
            return None

        # Use the role from the first piece (all pieces should have the same role)
        message_role: ChatMessageRole = first_piece.role

        return ChatMessageListDictContent(role=message_role, content=content_items).model_dump(exclude_none=True)

    def _create_system_message(self, request_pieces: Sequence[PromptRequestPiece]) -> Dict[str, Any]:
        """
        Create a system message from request pieces.

        Args:
            request_pieces: List of request pieces for the system message.

        Returns:
            A formatted system message dictionary.

        Raises:
            ValueError: If system message format is invalid.
        """
        if len(request_pieces) > 1:
            raise ValueError(
                f"System messages must contain exactly one request piece. Found {len(request_pieces)} pieces."
            )

        system_piece = request_pieces[0]
        system_role: ChatMessageRole = "system"
        return ChatMessage(role=system_role, content=system_piece.converted_value).model_dump(exclude_none=True)

    async def _build_content_items(self, request_pieces: Sequence[PromptRequestPiece]) -> List[Dict[str, Any]]:
        """
        Build content items from request pieces.

        Args:
            request_pieces: List of request pieces to process.

        Returns:
            List of formatted content items.
        """
        content_items: List[Dict[str, Any]] = []

        for piece in request_pieces:
            content_item = await self._create_content_item(piece)
            if content_item:  # Only add non-None items
                content_items.append(content_item)

        return content_items

    async def _create_content_item(self, piece: PromptRequestPiece) -> Optional[Dict[str, Any]]:
        """
        Create a content item from a single request piece.

        Args:
            piece: The PromptRequestPiece to convert.

        Returns:
            A formatted content item dictionary, or None for skipped types.

        Raises:
            ValueError: If the data type is not supported.
        """
        data_type = piece.converted_value_data_type

        if data_type == "text":
            return self._create_text_content_item(piece)
        elif data_type == "image_path":
            return await self._create_image_content_item(piece)
        elif data_type == "reasoning":
            # Reasoning summaries are intentionally not passed back to the target
            return None
        else:
            raise ValueError(f"Multimodal data type '{data_type}' is not yet supported for role '{piece.role}'")

    def _create_text_content_item(self, piece: PromptRequestPiece) -> Dict[str, Any]:
        """
        Create a text content item.

        Args:
            piece: The PromptRequestPiece containing text data.

        Returns:
            A formatted text content item.
        """
        content_type = "input_text" if piece.role == "user" else "output_text"
        return {"type": content_type, "text": piece.converted_value}

    async def _create_image_content_item(self, piece: PromptRequestPiece) -> Dict[str, Any]:
        """
        Create an image content item.

        Args:
            piece: The PromptRequestPiece containing image path data.

        Returns:
            A formatted image content item.

        Raises:
            ValueError: If image conversion fails.
        """
        try:
            data_base64_encoded_url = await convert_local_image_to_data_url(piece.converted_value)
            return {"type": "input_image", "image_url": {"url": data_base64_encoded_url}}
        except Exception as e:
            raise ValueError(f"Failed to convert image at path '{piece.converted_value}': {e}") from e

    def _translate_roles(self, conversation: List[Dict[str, Any]]) -> None:
        # The "system" role is mapped to "developer" in the OpenAI Response API.
        for request in conversation:
            if request.get("role") == "system":
                request["role"] = "developer"

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
            # json_schema not yet supported by PyRIT
            "text": {"format": {"type": "json_object"}} if is_json_response else None,
        }

        if self._extra_body_parameters:
            body_parameters.update(self._extra_body_parameters)

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
            response_start = response[:100]
            raise PyritException(
                message=f"Failed to parse response from model {self._model_name} at {self._endpoint} as JSON.\n"
                f"Response: {response_start}\nFull error: {e}"
            )

        status = response.get("status")
        error = response.get("error")

        # Handle error responses
        if status is None:
            if error and error.get("code", "") == "content_filter":
                # TODO validate that this is correct with AOAI
                # Content filter with status 200 indicates that the model output was filtered
                # https://learn.microsoft.com/en-us/azure/ai-services/openai/concepts/content-filter
                return handle_bad_request_exception(
                    response_text=open_ai_str_response, request=request_piece, error_code=200, is_content_filter=True
                )
            else:
                raise PyritException(message=f"Unexpected response format: {response}. Expected 'status' key.")
        elif status != "completed" or error is not None:
            raise PyritException(message=f"Status {status} and error {error} from response: {response}")

        # Extract response pieces from the response object
        extracted_response_pieces = []
        for section in response.get("output", []):
            piece = self._parse_response_output_section(section=section, request_piece=request_piece, error=error)

            if piece is None:
                continue

            extracted_response_pieces.append(piece)

        if not extracted_response_pieces:
            raise PyritException(message="No valid response pieces found in the response.")

        return PromptRequestResponse(request_pieces=extracted_response_pieces)

    def _parse_response_output_section(
        self, *, section: dict, request_piece: PromptRequestPiece, error: Optional[PromptResponseError]
    ) -> PromptRequestPiece | None:
        piece_type: PromptDataType = "text"
        section_type = section.get("type", "")

        if section_type == PromptRequestPieceType.MESSAGE:
            section_content = section.get("content", [])
            if len(section_content) == 0:
                raise EmptyResponseException(message="The chat returned an empty message section.")
            piece_value = section_content[0].get("text", "")
        elif section_type == PromptRequestPieceType.REASONING:
            piece_value = ""
            piece_type = "reasoning"
            for summary_piece in section.get("summary", []):
                if summary_piece.get("type", "") == "summary_text":
                    piece_value += summary_piece.get("text", "")
            if not piece_value:
                return None  # Skip empty reasoning summaries
        else:
            # other options include "image_generation_call", "file_search_call", "function_call",
            # "web_search_call", "computer_call", "code_interpreter_call", "local_shell_call",
            # "mcp_call", "mcp_list_tools", "mcp_approval_request"
            raise ValueError(
                f"Unsupported response type: {section_type}. PyRIT does not yet support this response type."
            )

        # Handle empty response
        if not piece_value:
            raise EmptyResponseException(message="The chat returned an empty response.")

        return PromptRequestPiece(
            role="assistant",
            original_value=piece_value,
            conversation_id=request_piece.conversation_id,
            labels=request_piece.labels,
            prompt_target_identifier=request_piece.prompt_target_identifier,
            orchestrator_identifier=request_piece.orchestrator_identifier,
            original_value_data_type=piece_type,
            response_error=error or "none",
        )

    def _validate_request(self, *, prompt_request: PromptRequestResponse) -> None:
        """Validates the structure and content of a prompt request for compatibility of this target.

        Args:
            prompt_request (PromptRequestResponse): The prompt request response object.

        Raises:
            ValueError: If any of the request pieces have a data type other than 'text' or 'image_path'.
        """

        converted_prompt_data_types = [
            request_piece.converted_value_data_type for request_piece in prompt_request.request_pieces
        ]

        # Some models may not support all of these
        for prompt_data_type in converted_prompt_data_types:
            if prompt_data_type not in ["text", "image_path"]:
                raise ValueError("This target only supports text and image_path.")
