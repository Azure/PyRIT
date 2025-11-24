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
    pyrit_target_retry,
)
from pyrit.models import (
    Message,
    MessagePiece,
    PromptDataType,
    PromptResponseError,
)
from pyrit.prompt_target import (
    OpenAITarget,
    PromptChatTarget,
    limit_requests_per_minute,
)

logger = logging.getLogger(__name__)


# Tool function registry (agentic extension)
ToolExecutor = Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]


class MessagePieceType(str, Enum):
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


class OpenAIResponseTarget(OpenAITarget, PromptChatTarget):
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
        super().__init__(**kwargs)

        # Validate temperature and top_p
        if temperature is not None and (temperature < 0 or temperature > 2):
            raise PyritException("temperature must be between 0 and 2 (inclusive).")
        if top_p is not None and (top_p < 0 or top_p > 1):
            raise PyritException("top_p must be between 0 and 1 (inclusive).")

        self._temperature = temperature
        self._top_p = top_p
        self._max_output_tokens = max_output_tokens

        # Accept both old Azure format (/responses) and new format (/openai/v1)
        # Accept base URLs (/v1), specific API paths (/responses), Azure formats
        response_url_patterns = [
            r"/v1$",
            r"/responses",
            r"/deployments/[^/]+/",
            r"openai/v1",
            r"\.models\.ai\.azure\.com",
        ]
        self._warn_if_irregular_endpoint(response_url_patterns)

        # Reasoning parameters are not yet supported by PyRIT.
        # See https://platform.openai.com/docs/api-reference/responses/create#responses-create-reasoning
        # for more information.

        self._extra_body_parameters = extra_body_parameters

        # Per-instance tool/func registries:
        self._custom_functions: Dict[str, ToolExecutor] = custom_functions or {}
        self._fail_on_missing_function: bool = fail_on_missing_function

        # Extract the grammar 'tool' if one is present
        # See
        # https://platform.openai.com/docs/guides/function-calling#context-free-grammars
        self._grammar_name: str | None = None
        if extra_body_parameters:
            tools = extra_body_parameters.get("tools", [])
            for tool in tools:
                if tool.get("type") == "custom" and tool.get("format", {}).get("type") == "grammar":
                    if self._grammar_name is not None:
                        raise ValueError("Multiple grammar tools detected; only one is supported.")
                    tool_name = tool.get("name")
                    logger.debug("Detected grammar tool: %s", tool_name)
                    self._grammar_name = tool_name

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

    async def _construct_input_item_from_piece(self, piece: MessagePiece) -> Dict[str, Any]:
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
            return {
                "type": "input_text" if piece.role in ["developer", "user"] else "output_text",
                "text": piece.converted_value,
            }
        if piece.converted_value_data_type == "image_path":
            data_url = await convert_local_image_to_data_url(piece.converted_value)
            return {"type": "input_image", "image_url": {"url": data_url}}
        raise ValueError(f"Unsupported piece type for inline content: {piece.converted_value_data_type}")

    async def _build_input_for_multi_modal_async(self, conversation: MutableSequence[Message]) -> List[Dict[str, Any]]:
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
            pieces = message.message_pieces
            if not pieces:
                raise ValueError(
                    f"Failed to process conversation message at index {msg_idx}: Message contains no message pieces"
                )

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

    async def _construct_request_body(self, conversation: MutableSequence[Message], is_json_response: bool) -> dict:
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

    async def _send_prompt_and_find_tool_call_async(
        self,
        *,
        message: Message,
    ) -> tuple[Message, Optional[dict[str, Any]]]:
        """
        Send the prompt and return the assistant message and any pending tool call.

        Returns:
            Tuple of (assistant_message, tool_call_dict_or_none)
        """
        # Core send logic with error handling
        self._validate_request(message=message)
        message_piece: MessagePiece = message.message_pieces[0]
        is_json_response = self.is_response_format_json(message_piece)

        conv = self._memory.get_conversation(conversation_id=message_piece.conversation_id)
        conv.append(message)
        logger.info(f"Sending the following prompt to the prompt target: {message}")

        body = await self._construct_request_body(conversation=conv, is_json_response=is_json_response)

        # Use unified error handling - automatically detects Response and validates
        result = await self._handle_openai_request(
            api_call=lambda: self._async_client.responses.create(**body),
            request=message,
        )

        # Append the result to memory conversation
        conv.append(result)

        # Extract tool call if present
        tool_call = self._find_last_pending_tool_call(result)
        return result, tool_call

    def _check_content_filter(self, response: Any) -> bool:
        """
        Check if a Response API response has a content filter error.

        Args:
            response: A Response object from the OpenAI SDK.

        Returns:
            True if content was filtered, False otherwise.
        """
        try:
            if hasattr(response, "error") and response.error is not None:
                return response.error.code == "content_filter"
        except (AttributeError, TypeError):
            pass
        return False

    def _validate_response(self, response: Any, request: MessagePiece) -> Optional[Message]:
        """
        Validate a Response API response for errors.

        Checks for:
        - Error responses (excluding content filtering which is checked separately)
        - Invalid status
        - Empty output

        Args:
            response: The Response object from the OpenAI SDK.
            request: The original request MessagePiece.

        Returns:
            None if valid, does not return Message for content filter (handled by _check_content_filter).

        Raises:
            PyritException: For unexpected response structures or errors.
            EmptyResponseException: When the API returns no valid output.
        """
        from pyrit.exceptions import EmptyResponseException, PyritException

        # Check for error response - error is a ResponseError object or None
        # (content_filter is handled by _check_content_filter)
        if response.error is not None and response.error.code != "content_filter":
            raise PyritException(message=f"Response error: {response.error.code} - {response.error.message}")

        # Check status - should be "completed" for successful responses
        if response.status != "completed":
            raise PyritException(message=f"Unexpected status: {response.status}")

        # Check for empty output
        if not response.output:
            logger.error("The response returned no valid output.")
            raise EmptyResponseException(message="The response returned an empty response.")

        return None

    async def _construct_message_from_response(self, response: Any, request: MessagePiece) -> Message:
        """
        Construct a Message from a Response API response.

        Args:
            response: The Response object from OpenAI SDK.
            request: The original request MessagePiece.

        Returns:
            Message: Constructed message with extracted content from output sections.
        """
        # Extract and parse message pieces from validated output sections
        extracted_response_pieces: List[MessagePiece] = []
        for section in response.output:
            piece = self._parse_response_output_section(
                section=section,
                message_piece=request,
                error=None,  # error is already handled in validation
            )
            if piece is None:
                continue
            extracted_response_pieces.append(piece)

        return Message(message_pieces=extracted_response_pieces)

    @limit_requests_per_minute
    @limit_requests_per_minute
    @pyrit_target_retry
    async def send_prompt_async(self, *, message: Message) -> Message:
        """
        Send prompt, handle agentic tool calls (function_call), return assistant output.

        Args:
            message: The initial prompt from the user.

        Returns:
            The final Message with the assistant's answer.
        """
        assistant_message, tool_call_section = await self._send_prompt_and_find_tool_call_async(message=message)

        while tool_call_section:
            # Execute the tool/function
            tool_output = await self._execute_call_section(tool_call_section)

            # Add the tool result as a tool message to the conversation
            # NOTE: Responses API expects a top-level {type:function_call_output, call_id, output}
            # Use the first piece from the original message as reference for conversation context
            reference_piece = message.message_pieces[0]
            tool_message = self._make_tool_message(
                tool_output, tool_call_section["call_id"], reference_piece=reference_piece
            )

            # Build combined message with all conversation history for next request
            # Get full conversation from memory
            conv = self._memory.get_conversation(conversation_id=reference_piece.conversation_id)
            conv.append(tool_message)

            merged: List[MessagePiece] = []
            for msg in conv:
                merged.extend(msg.message_pieces)

            # TODO: There is likely a bug here; there are different roles in a single response??
            message = Message(message_pieces=merged, skip_validation=True)

            # Send again and check for another tool call
            assistant_message, tool_call_section = await self._send_prompt_and_find_tool_call_async(message=message)

        # No other tool call found, return the final assistant message
        return assistant_message

    def is_json_response_supported(self) -> bool:
        """Indicates that this target supports JSON response format."""
        return True

    def _parse_response_output_section(
        self, *, section, message_piece: MessagePiece, error: Optional[PromptResponseError]
    ) -> MessagePiece | None:
        """
        Parse model output sections, forwarding tool-calls for the agentic loop.

        Args:
            section: The section object from OpenAI SDK (Pydantic model).
            message_piece: The original message piece.
            error: Any error information from OpenAI.

        Returns:
            A MessagePiece for this section, or None to skip.
        """
        section_type = section.type
        piece_type: PromptDataType = "text"  # Default, always set!
        piece_value = ""

        if section_type == MessagePieceType.MESSAGE:
            section_content = section.content
            if len(section_content) == 0:
                raise EmptyResponseException(message="The chat returned an empty message section.")
            piece_value = section_content[0].text

        elif section_type == MessagePieceType.REASONING:
            # Keep the full reasoning JSON as a piece (internal use / debugging)
            piece_value = json.dumps(section.model_dump(), separators=(",", ":"))
            piece_type = "reasoning"

        elif section_type == MessagePieceType.FUNCTION_CALL:
            # Forward the tool call verbatim so the agentic loop can execute it
            piece_value = json.dumps(section.model_dump(), separators=(",", ":"))
            piece_type = "function_call"

        elif section_type == MessagePieceType.WEB_SEARCH_CALL:
            # Forward web_search_call verbatim as a tool_call
            piece_value = json.dumps(section.model_dump(), separators=(",", ":"))
            piece_type = "tool_call"

        elif section_type == "custom_tool_call":
            # Had a Lark grammar (hopefully)
            # See
            # https://platform.openai.com/docs/guides/function-calling#context-free-grammars
            logger.debug("Detected custom_tool_call in response, assuming grammar constraint.")
            extracted_grammar_name = section.name
            if extracted_grammar_name != self._grammar_name:
                msg = "Mismatched grammar name in custom_tool_call "
                msg += f"(expected {self._grammar_name}, got {extracted_grammar_name})"
                logger.error(msg)
                raise ValueError(msg)
            piece_value = section.input
            if len(piece_value) == 0:
                raise EmptyResponseException(message="The chat returned an empty message section.")

        else:
            # Other possible types are not yet handled in PyRIT
            return None

        # Handle empty response
        if not piece_value:
            raise EmptyResponseException(message="The chat returned an empty response.")

        return MessagePiece(
            role="assistant",
            original_value=piece_value,
            conversation_id=message_piece.conversation_id,
            labels=message_piece.labels,
            prompt_target_identifier=message_piece.prompt_target_identifier,
            attack_identifier=message_piece.attack_identifier,
            original_value_data_type=piece_type,
            response_error=error or "none",
        )

    def _validate_request(self, *, message: Message) -> None:
        """
        Validates the structure and content of a message for compatibility of this target.

        Args:
            message (Message): The message object.

        Raises:
            ValueError: If any of the message pieces have a data type other than supported set.
        """
        # Some models may not support all of these; we accept them at the transport layer
        # so the Responses API can decide. We include reasoning and function_call_output now.
        allowed_types = {"text", "image_path", "function_call", "tool_call", "function_call_output", "reasoning"}
        for message_piece in message.message_pieces:
            if message_piece.converted_value_data_type not in allowed_types:
                raise ValueError(f"Unsupported data type: {message_piece.converted_value_data_type}")
        return

    # Agentic helpers (module scope)

    def _find_last_pending_tool_call(self, reply: Message) -> Optional[dict[str, Any]]:
        """
        Return the last tool-call section in assistant messages, or None.
        Looks for a piece whose value parses as JSON with a 'type' key matching function_call.
        """
        for piece in reversed(reply.message_pieces):
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
        if not name:
            if self._fail_on_missing_function:
                raise ValueError("Function call section missing 'name' field")
            return {
                "error": "missing_function_name",
                "tool_call_section": tool_call_section,
            }

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

    def _make_tool_message(self, output: dict[str, Any], call_id: str, *, reference_piece: MessagePiece) -> Message:
        """
        Wrap tool output as a top-level function_call_output artifact.

        Args:
            output: The tool output to wrap.
            call_id: The call ID for the function call.
            reference_piece: A reference piece to copy conversation context from.

        Returns:
            A Message containing the function call output.
        """
        output_str = output if isinstance(output, str) else json.dumps(output, separators=(",", ":"))
        piece = MessagePiece(
            role="assistant",
            original_value=json.dumps(
                {"type": "function_call_output", "call_id": call_id, "output": output_str},
                separators=(",", ":"),
            ),
            original_value_data_type="function_call_output",
            conversation_id=reference_piece.conversation_id,
            labels={"call_id": call_id},
            prompt_target_identifier=reference_piece.prompt_target_identifier,
            attack_identifier=reference_piece.attack_identifier,
        )

        return Message(message_pieces=[piece])
