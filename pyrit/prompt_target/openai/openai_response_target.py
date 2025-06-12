# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import json
import logging
from typing import Any, Dict, List, MutableSequence, Optional

from pyrit.common import convert_local_image_to_data_url
from pyrit.exceptions import (
    EmptyResponseException,
    PyritException,
    handle_bad_request_exception,
)
from pyrit.models import (
    ChatMessage,
    ChatMessageListDictContent,
    PromptDataType,
    PromptRequestPiece,
    PromptRequestResponse,
    PromptResponseError,
)
from pyrit.prompt_target.openai.openai_chat_target_base import OpenAIChatTargetBase

logger = logging.getLogger(__name__)


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
            extra_body_parameters (dict, Optional): Additional parameters to be included in the request body.
            httpx_client_kwargs (dict, Optional): Additional kwargs to be passed to the
                httpx.AsyncClient() constructor.
                For example, to specify a 3 minutes timeout: httpx_client_kwargs={"timeout": 180}
        """
        super().__init__(
            api_version=api_version, temperature=temperature, top_p=top_p, is_json_supported=is_json_supported, **kwargs
        )

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
            conversation (list[PromptRequestResponse]): A list of PromptRequestResponse objects.

        Returns:
            list[dict]: The list of constructed chat messages.
        """
        full_request: List[Dict[str, Any]] = []
        for message in conversation:

            prompt_request_pieces = message.request_pieces

            if len(prompt_request_pieces) == 0:
                raise ValueError("No prompt request pieces found in the conversation.")

            # System messages are a special case. Instead of the nested formatting with
            # "type" and "text" at a deeper level, they are just a single content string.
            first_piece = prompt_request_pieces[0]
            if first_piece.role == "system":
                if len(prompt_request_pieces) > 1:
                    raise ValueError(
                        "System messages should only have a single request piece. "
                        "Multiple system messages are not supported."
                    )
                full_request.append(
                    ChatMessage(role="system", content=first_piece.converted_value).model_dump(exclude_none=True)
                )
                continue

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
                    data_base64_encoded_url = await convert_local_image_to_data_url(
                        prompt_request_piece.converted_value
                    )
                    image_url_entry = {"url": data_base64_encoded_url}
                    entry = {"type": "input_image", "image_url": image_url_entry}  # type: ignore
                    content.append(entry)
                elif prompt_request_piece.converted_value_data_type == "reasoning":
                    # reasoning summaries are not passed back to the target
                    pass
                else:
                    raise ValueError(
                        f"Multimodal data type {prompt_request_piece.converted_value_data_type} is not yet supported."
                    )

            full_request.append(
                ChatMessageListDictContent(role=role, content=content).model_dump(exclude_none=True)  # type: ignore
            )

        self._translate_roles(conversation=full_request)

        return full_request

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

        response = ""
        try:
            response = json.loads(open_ai_str_response)
        except json.JSONDecodeError as e:
            response_start = response[:100]
            raise PyritException(
                message=f"Failed to parse response from model {self._model_name} at {self._endpoint} as JSON.\nResponse: {response_start}\nFull error: {e}"
            )

        status = response.get("status")
        error = response.get("error")

        # Handle error responses
        if status is None:
            if error and "code" in error and error["code"] == "content_filter":
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
        if section_type == "message":
            section_content = section.get("content", [])
            if len(section_content) == 0:
                raise EmptyResponseException(message="The chat returned an empty message section.")

            piece_value = section_content[0].get("text", "")
        elif section_type == "reasoning":
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
