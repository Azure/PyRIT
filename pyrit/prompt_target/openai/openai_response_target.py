# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import json
import logging
from enum import Enum
from typing import (
    Any,
    Awaitable,
    Callable,
    Dict,
    List,
    MutableSequence,
    Optional,
)

from pyrit.common import convert_local_image_to_data_url
from pyrit.exceptions import (
    EmptyResponseException,
    PyritException,
    handle_bad_request_exception,
)
from pyrit.models import (
    PromptDataType,
    PromptRequestPiece,
    PromptRequestResponse,
    PromptResponseError,
)
from pyrit.prompt_target.openai.openai_chat_target_base import OpenAIChatTargetBase

logger = logging.getLogger(__name__)


# Tool function registry (agentic extension)
ToolExecutor = Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]


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
        custom_functions: Optional[Dict[str, ToolExecutor]] = None,
        api_version: Optional[str] = "2025-03-01-preview",
        max_output_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        extra_body_parameters: Optional[dict[str, Any]] = None,
        fail_on_missing_function: bool = False,
        **kwargs,
    ):
        """
        Initializes the OpenAIResponseTarget with the provided parameters.

        Args:
            custom_functions: Mapping of user-defined function names (e.g., "my_func").
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
            fail_on_missing_function: if True, raise when a function_call references
                an unknown function or does not output a function; if False, return a structured error so we can
                wrap it as function_call_output and let the model potentially recover
                (e.g., pick another tool or ask for clarification).
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

        # Per-instance tool/func registries:
        self._custom_functions: Dict[str, ToolExecutor] = custom_functions or {}
        self._fail_on_missing_function: bool = fail_on_missing_function

    def _set_openai_env_configuration_vars(self) -> None:
        self.model_name_environment_variable = "OPENAI_RESPONSES_MODEL"
        self.endpoint_environment_variable = "OPENAI_RESPONSES_ENDPOINT"
        self.api_key_environment_variable = "OPENAI_RESPONSES_KEY"
        return

    # Helpers kept on the class for reuse + testability
    def _flush_message(self, role: Optional[str], content: List[Dict[str, Any]], output: List[Dict[str, Any]]) -> None:
        """
        Append a role message and clear the working buffer.

        Args:
            role: Role to emit ("user" / "assistant" / "system").
            content: Accumulated content items for the role.
            output: Destination list to append the message to. It holds a list of dicts containing
                key-value pairs representing the role and content.

        Returns:
            None. Mutates `output` (append) and `content` (clear).
        """
        if role and content:
            output.append({"role": role, "content": list(content)})
            content.clear()
        return

    async def _construct_input_item_from_piece(self, piece: PromptRequestPiece) -> Dict[str, Any]:
        """
        Convert a single inline piece into a Responses API content item.

        Args:
            piece: The inline piece (text or image_path).

        Returns:
            A dict in the Responses API content item shape.

        Raises:
            ValueError: If the piece type is not supported for inline content. Supported types are text and
                image paths.
        """
        if piece.converted_value_data_type == "text":
            return {"type": "input_text" if piece.role == "user" else "output_text", "text": piece.converted_value}
        if piece.converted_value_data_type == "image_path":
            data_url = await convert_local_image_to_data_url(piece.converted_value)
            return {"type": "input_image", "image_url": {"url": data_url}}
        raise ValueError(f"Unsupported piece type for inline content: {piece.converted_value_data_type}")

    async def _build_input_for_multi_modal_async(
        self, conversation: MutableSequence[PromptRequestResponse]
    ) -> List[Dict[str, Any]]:
        """
        Build the Responses API `input` array.

        Groups inline content (text/images) into role messages and emits tool artifacts
        (reasoning, function_call, function_call_output, web_search_call, etc.) as top-level
        items — per the Responses API schema.

        Args:
            conversation: Ordered list of user/assistant/tool artifacts to serialize.

        Returns:
            A list of input items ready for the Responses API.

        Raises:
            ValueError: If the conversation is empty or a system message has >1 piece.
        """
        if not conversation:
            raise ValueError("Conversation cannot be empty")

        input_items: List[Dict[str, Any]] = []

        for msg_idx, message in enumerate(conversation):
            pieces = message.request_pieces
            if not pieces:
                continue

            # System message -> single role message (remapped to developer later)
            if pieces[0].role == "system":
                if len(pieces) != 1:
                    raise ValueError("System messages must have exactly one piece.")
                input_items.append(
                    {
                        "role": "system",
                        "content": [{"type": "input_text", "text": pieces[0].converted_value}],
                    }
                )
                continue

            role: Optional[str] = None
            content: List[Dict[str, Any]] = []

            for piece in pieces:
                dtype = piece.converted_value_data_type

                # Inline, role-batched content
                if dtype in {"text", "image_path"}:
                    if role is None:
                        role = piece.role
                    elif piece.role != role:
                        self._flush_message(role, content, input_items)
                        role = piece.role

                    content.append(await self._construct_input_item_from_piece(piece))
                    continue

                # Top-level artifacts (flush any pending role content first)
                self._flush_message(role, content, input_items)
                role = None

                if dtype not in {"reasoning", "function_call", "function_call_output", "tool_call"}:
                    raise ValueError(f"Unsupported data type '{dtype}' in message index {msg_idx}")

                if dtype in {"reasoning", "function_call", "tool_call"}:
                    # Already in API shape in original_value
                    input_items.append(json.loads(piece.original_value))

                if dtype == "function_call_output":
                    payload = json.loads(piece.original_value)
                    output = payload.get("output")
                    if not isinstance(output, str):
                        # Responses API requires string output; serialize if needed
                        output = json.dumps(output, separators=(",", ":"))
                    input_items.append(
                        {
                            "type": "function_call_output",
                            "call_id": payload["call_id"],
                            "output": output,
                        }
                    )

            # Flush trailing role content for this message
            self._flush_message(role, content, input_items)

        # Responses API maps system -> developer
        self._translate_roles(conversation=input_items)
        return input_items

    def _translate_roles(self, conversation: List[Dict[str, Any]]) -> None:
        # The "system" role is mapped to "developer" in the OpenAI Response API.
        for request in conversation:
            if request.get("role") == "system":
                request["role"] = "developer"
        return

    async def _construct_request_body(
        self, conversation: MutableSequence[PromptRequestResponse], is_json_response: bool
    ) -> dict:
        """
        Construct the request body to send to the Responses API.

        NOTE: The Responses API uses top-level `response_format` for JSON,
        not `text.format` from the old Chat Completions style.
        """
        input_items = await self._build_input_for_multi_modal_async(conversation)

        body_parameters = {
            "model": self._model_name,
            "max_output_tokens": self._max_output_tokens,
            "temperature": self._temperature,
            "top_p": self._top_p,
            "stream": False,
            "input": input_items,
            # Correct JSON response format per Responses API
            "response_format": {"type": "json_object"} if is_json_response else None,
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
        """
        Parse the Responses API JSON into internal PromptRequestResponse.
        """
        response: dict[str, Any]
        try:
            response = json.loads(open_ai_str_response)
        except json.JSONDecodeError as e:
            response_start = open_ai_str_response[:100]
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
        extracted_response_pieces: List[PromptRequestPiece] = []
        for section in response.get("output", []):
            piece = self._parse_response_output_section(section=section, request_piece=request_piece, error=error)
            if piece is None:
                continue
            extracted_response_pieces.append(piece)

        if not extracted_response_pieces:
            raise PyritException(message="No valid response pieces found in the response.")

        return PromptRequestResponse(request_pieces=extracted_response_pieces)

    async def send_prompt_async(self, *, prompt_request: PromptRequestResponse) -> PromptRequestResponse:
        """
        Send prompt, handle agentic tool calls (function_call), return assistant output.

        Args:
            prompt_request: The initial prompt from the user.

        Returns:
            The final PromptRequestResponse with the assistant's answer.
        """
        conversation: MutableSequence[PromptRequestResponse] = [prompt_request]
        send_prompt_async = super().send_prompt_async  # bind for inner function

        async def _send_prompt_and_find_tool_call_async(
            prompt_request: PromptRequestResponse,
        ) -> Optional[dict[str, Any]]:
            """Send the prompt and return the last pending tool call, if any."""
            assistant_reply = await send_prompt_async(prompt_request=prompt_request)
            conversation.append(assistant_reply)
            return self._find_last_pending_tool_call(assistant_reply)

        tool_call_section = await _send_prompt_and_find_tool_call_async(prompt_request=prompt_request)
        while tool_call_section:
            # Execute the tool/function
            tool_output = await self._execute_call_section(tool_call_section)

            # Add the tool result as a tool message to the conversation
            # NOTE: Responses API expects a top-level {type:function_call_output, call_id, output}
            tool_message = self._make_tool_message(tool_output, tool_call_section["call_id"])
            conversation.append(tool_message)

            # Re-ask with combined history (user + function_call + function_call_output)
            merged: List[PromptRequestPiece] = []
            for msg in conversation:
                merged.extend(msg.request_pieces)
            prompt_request = PromptRequestResponse(request_pieces=merged)

            # Send again and check for another tool call
            tool_call_section = await _send_prompt_and_find_tool_call_async(prompt_request=prompt_request)

        # No other tool call found, so assistant message is complete and return last assistant reply!
        return conversation[-1]

    def _parse_response_output_section(
        self, *, section: dict, request_piece: PromptRequestPiece, error: Optional[PromptResponseError]
    ) -> PromptRequestPiece | None:
        """
        Parse model output sections, forwarding tool-calls for the agentic loop.

        Args:
            section: The section dict from OpenAI output.
            request_piece: The original request piece.
            error: Any error information from OpenAI.

        Returns:
            A PromptRequestPiece for this section, or None to skip.
        """
        section_type = section.get("type", "")
        piece_type: PromptDataType = "text"  # Default, always set!
        piece_value = ""

        if section_type == PromptRequestPieceType.MESSAGE:
            section_content = section.get("content", [])
            if len(section_content) == 0:
                raise EmptyResponseException(message="The chat returned an empty message section.")
            piece_value = section_content[0].get("text", "")

        elif section_type == PromptRequestPieceType.REASONING:
            # Keep the full reasoning JSON as a piece (internal use / debugging)
            piece_value = json.dumps(section, separators=(",", ":"))
            piece_type = "reasoning"

        elif section_type == PromptRequestPieceType.FUNCTION_CALL:
            # Forward the tool call verbatim so the agentic loop can execute it
            piece_value = json.dumps(section, separators=(",", ":"))
            piece_type = "function_call"

        elif section_type == PromptRequestPieceType.WEB_SEARCH_CALL:
            # Forward web_search_call verbatim as a tool_call
            piece_value = json.dumps(section, separators=(",", ":"))
            piece_type = "tool_call"

        else:
            # Other possible types are not yet handled in PyRIT
            return None

        # Handle empty response
        if not piece_value:
            raise EmptyResponseException(message="The chat returned an empty response.")

        return PromptRequestPiece(
            role="assistant",
            original_value=piece_value,
            conversation_id=request_piece.conversation_id,
            labels=request_piece.labels,
            prompt_target_identifier=request_piece.prompt_target_identifier,
            attack_identifier=request_piece.attack_identifier,
            original_value_data_type=piece_type,
            response_error=error or "none",
        )

    def _validate_request(self, *, prompt_request: PromptRequestResponse) -> None:
        """Validates the structure and content of a prompt request for compatibility of this target.

        Args:
            prompt_request (PromptRequestResponse): The prompt request response object.

        Raises:
            ValueError: If any of the request pieces have a data type other than supported set.
        """
        # Some models may not support all of these; we accept them at the transport layer
        # so the Responses API can decide. We include reasoning and function_call_output now.
        allowed_types = {"text", "image_path", "function_call", "tool_call", "function_call_output", "reasoning"}
        for request_piece in prompt_request.request_pieces:
            if request_piece.converted_value_data_type not in allowed_types:
                raise ValueError(f"Unsupported data type: {request_piece.converted_value_data_type}")
        return

    # Agentic helpers (module scope)

    def _find_last_pending_tool_call(self, reply: PromptRequestResponse) -> Optional[dict[str, Any]]:
        """
        Return the last tool-call section in assistant messages, or None.
        Looks for a piece whose value parses as JSON with a 'type' key matching function_call.
        """
        for piece in reversed(reply.request_pieces):
            if piece.role == "assistant":
                try:
                    section = json.loads(piece.original_value)
                except Exception:
                    continue
                if section.get("type") == "function_call":
                    # Do NOT skip function_call even if status == "completed" — we still need to emit the output.
                    return section
        return None

    async def _execute_call_section(self, tool_call_section: dict[str, Any]) -> dict[str, Any]:
        """
        Execute a function_call from the custom_functions registry.

        Returns:
            A dict payload (will be serialized and sent as function_call_output).
            If fail_on_missing_function=False and a function is missing or no function is not called, returns:
            {"error": "function_not_found", "missing_function": "<name>", "available_functions": [...]}
        """
        name = tool_call_section.get("name")
        args_json = tool_call_section.get("arguments", "{}")
        try:
            args = json.loads(args_json)
        except Exception:
            # If arguments are not valid JSON, surface a structured error (or raise)
            if self._fail_on_missing_function:
                raise ValueError(f"Malformed arguments for function '{name}': {args_json}")
            logger.warning("Malformed arguments for function '%s': %s", name, args_json)
            return {
                "error": "malformed_arguments",
                "function": name,
                "raw_arguments": args_json,
            }

        fn = self._custom_functions.get(name)
        if fn is None:
            if self._fail_on_missing_function:
                raise KeyError(f"Function '{name}' is not registered")
            # Tolerant mode: return a structured error so we can wrap it as function_call_output
            available = sorted(self._custom_functions.keys())
            logger.warning("Function '%s' not registered. Available: %s", name, available)
            return {
                "error": "function_not_found",
                "missing_function": name,
                "available_functions": available,
            }

        return await fn(args)

    def _make_tool_message(self, output: dict[str, Any], call_id: str) -> PromptRequestResponse:
        """
        Wrap tool output as a top-level function_call_output artifact.

        The Responses API requires a string in the "output" field; we serialize objects.
        """
        output_str = output if isinstance(output, str) else json.dumps(output, separators=(",", ":"))
        piece = PromptRequestPiece(
            role="assistant",
            original_value=json.dumps(
                {"type": "function_call_output", "call_id": call_id, "output": output_str},
                separators=(",", ":"),
            ),
            original_value_data_type="function_call_output",
            labels={"call_id": call_id},
        )
        return PromptRequestResponse(request_pieces=[piece])
