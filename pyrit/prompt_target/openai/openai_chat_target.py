# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import json
import logging
from typing import Any, MutableSequence, Optional

from pyrit.common import convert_local_image_to_data_url
from pyrit.exceptions import (
    EmptyResponseException,
    PyritException,
    handle_bad_request_exception,
)
from pyrit.models import (
    ChatMessageListDictContent,
    PromptRequestPiece,
    PromptRequestResponse,
    construct_response_from_request,
)
from pyrit.models.chat_message import ChatMessage
from pyrit.prompt_target.openai.openai_chat_target_base import OpenAIChatTargetBase

logger = logging.getLogger(__name__)


class OpenAIChatTarget(OpenAIChatTargetBase):
    """
    This class facilitates multimodal (image and text) input and text output generation

    This works with GPT3.5, GPT4, GPT4o, GPT-V, and other compatible models
    """

    def __init__(
        self,
        *,
        max_completion_tokens: Optional[int] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
        presence_penalty: Optional[float] = None,
        seed: Optional[int] = None,
        n: Optional[int] = None,
        is_json_supported: bool = True,
        extra_body_parameters: Optional[dict[str, Any]] = None,
        **kwargs,
    ):
        """
        Args:
            model_name (str, Optional): The name of the model.
            endpoint (str, Optional): The target URL for the OpenAI service.
            api_key (str, Optional): The API key for accessing the Azure OpenAI service.
                Defaults to the OPENAI_CHAT_KEY environment variable.
            headers (str, Optional): Headers of the endpoint (JSON).
            use_aad_auth (bool, Optional): When set to True, user authentication is used
                instead of API Key. DefaultAzureCredential is taken for
                https://cognitiveservices.azure.com/.default . Please run `az login` locally
                to leverage user AuthN.
            api_version (str, Optional): The version of the Azure OpenAI API. Defaults to
                "2024-10-21".
            max_requests_per_minute (int, Optional): Number of requests the target can handle per
                minute before hitting a rate limit. The number of requests sent to the target
                will be capped at the value provided.
            max_completion_tokens (int, Optional): An upper bound for the number of tokens that
                can be generated for a completion, including visible output tokens and
                reasoning tokens.

                NOTE: Specify this value when using an o1 series model.
            max_tokens (int, Optional): The maximum number of tokens that can be
                generated in the chat completion. This value can be used to control
                costs for text generated via API.

                This value is now deprecated in favor of `max_completion_tokens`, and IS NOT
                COMPATIBLE with o1 series models.
            temperature (float, Optional): The temperature parameter for controlling the
                randomness of the response.
            top_p (float, Optional): The top-p parameter for controlling the diversity of the
                response.
            frequency_penalty (float, Optional): The frequency penalty parameter for penalizing
                frequently generated tokens.
            presence_penalty (float, Optional): The presence penalty parameter for penalizing
                tokens that are already present in the conversation history.
            seed (int, Optional): If specified, openAI will make a best effort to sample deterministically,
                such that repeated requests with the same seed and parameters should return the same result.
            n (int, Optional): The number of completions to generate for each prompt.
            is_json_supported (bool, Optional): If True, the target will supports formatting responses as JSON by
                setting the response_format header. Official OpenAI models all support this, but if you are using
                this target with different models, is_json_supported should be set correctly to avoid issues when
                using adversarial infrastructure (e.g. Crescendo scorers will set this flag).
            extra_body_parameters (dict, Optional): Additional parameters to be included in the request body.
            httpx_client_kwargs (dict, Optional): Additional kwargs to be passed to the
                httpx.AsyncClient() constructor.
                For example, to specify a 3 minutes timeout: httpx_client_kwargs={"timeout": 180}
        """
        super().__init__(temperature=temperature, top_p=top_p, **kwargs)

        if max_completion_tokens and max_tokens:
            raise ValueError("Cannot provide both max_tokens and max_completion_tokens.")

        self._max_completion_tokens = max_completion_tokens
        self._max_tokens = max_tokens
        self._frequency_penalty = frequency_penalty
        self._presence_penalty = presence_penalty
        self._seed = seed
        self._n = n
        self._extra_body_parameters = extra_body_parameters

    def _set_openai_env_configuration_vars(self) -> None:
        self.model_name_environment_variable = "OPENAI_CHAT_MODEL"
        self.endpoint_environment_variable = "OPENAI_CHAT_ENDPOINT"
        self.api_key_environment_variable = "OPENAI_CHAT_KEY"

    async def _build_chat_messages_async(self, conversation: MutableSequence[PromptRequestResponse]) -> list[dict]:
        """Builds chat messages based on prompt request response entries.

        Args:
            conversation (list[PromptRequestResponse]): A list of PromptRequestResponse objects.

        Returns:
            list[dict]: The list of constructed chat messages.
        """
        if self._is_text_message_format(conversation):
            return self._build_chat_messages_for_text(conversation)
        else:
            return await self._build_chat_messages_for_multi_modal_async(conversation)

    def _is_text_message_format(self, conversation: MutableSequence[PromptRequestResponse]) -> bool:
        """Checks if the request piece is in text message format.

        Args:
            conversation list[PromptRequestResponse]: The conversation

        Returns:
            bool: True if the request piece is in text message format, False otherwise.
        """
        for turn in conversation:
            if len(turn.request_pieces) != 1:
                return False
            if turn.request_pieces[0].converted_value_data_type != "text":
                return False
        return True

    def _build_chat_messages_for_text(self, conversation: MutableSequence[PromptRequestResponse]) -> list[dict]:
        """
        Builds chat messages based on prompt request response entries. This is needed because many
        openai "compatible" models don't support ChatMessageListDictContent format (this is more univerally accepted)

        Args:
            conversation (list[PromptRequestResponse]): A list of PromptRequestResponse objects.

        Returns:
            list[dict]: The list of constructed chat messages.
        """
        chat_messages: list[dict] = []
        for prompt_req_resp_entry in conversation:
            # validated to only have one text entry

            if len(prompt_req_resp_entry.request_pieces) != 1:
                raise ValueError("_build_chat_messages_for_text only supports a single prompt request piece.")

            prompt_request_piece = prompt_req_resp_entry.request_pieces[0]

            if prompt_request_piece.converted_value_data_type != "text":
                raise ValueError("_build_chat_messages_for_text only supports text.")

            message = ChatMessage(role=prompt_request_piece.role, content=prompt_request_piece.converted_value)
            chat_messages.append(message.model_dump(exclude_none=True))

        return chat_messages

    async def _build_chat_messages_for_multi_modal_async(
        self, conversation: MutableSequence[PromptRequestResponse]
    ) -> list[dict]:
        """
        Builds chat messages based on prompt request response entries.

        Args:
            conversation (list[PromptRequestResponse]): A list of PromptRequestResponse objects.

        Returns:
            list[dict]: The list of constructed chat messages.
        """
        chat_messages: list[dict] = []
        for prompt_req_resp_entry in conversation:
            prompt_request_pieces = prompt_req_resp_entry.request_pieces

            content = []
            role = None
            for prompt_request_piece in prompt_request_pieces:
                role = prompt_request_piece.role
                if prompt_request_piece.converted_value_data_type == "text":
                    entry = {"type": "text", "text": prompt_request_piece.converted_value}
                    content.append(entry)
                elif prompt_request_piece.converted_value_data_type == "image_path":
                    data_base64_encoded_url = await convert_local_image_to_data_url(
                        prompt_request_piece.converted_value
                    )
                    image_url_entry = {"url": data_base64_encoded_url}
                    entry = {"type": "image_url", "image_url": image_url_entry}  # type: ignore
                    content.append(entry)
                else:
                    raise ValueError(
                        f"Multimodal data type {prompt_request_piece.converted_value_data_type} is not yet supported."
                    )

            if not role:
                raise ValueError("No role could be determined from the prompt request pieces.")

            chat_message = ChatMessageListDictContent(role=role, content=content)  # type: ignore
            chat_messages.append(chat_message.model_dump(exclude_none=True))
        return chat_messages

    async def _construct_request_body(
        self, conversation: MutableSequence[PromptRequestResponse], is_json_response: bool
    ) -> dict:
        messages = await self._build_chat_messages_async(conversation)

        body_parameters = {
            "model": self._model_name,
            "max_completion_tokens": self._max_completion_tokens,
            "max_tokens": self._max_tokens,
            "temperature": self._temperature,
            "top_p": self._top_p,
            "frequency_penalty": self._frequency_penalty,
            "presence_penalty": self._presence_penalty,
            "stream": False,
            "seed": self._seed,
            "n": self._n,
            "messages": messages,
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

        finish_reason = response["choices"][0]["finish_reason"]
        extracted_response: str = ""
        # finish_reason="stop" means API returned complete message and
        # "length" means API returned incomplete message due to max_tokens limit.
        if finish_reason in ["stop", "length"]:
            extracted_response = response["choices"][0]["message"]["content"]

            # Handle empty response
            if not extracted_response:
                logger.log(logging.ERROR, "The chat returned an empty response.")
                raise EmptyResponseException(message="The chat returned an empty response.")
        elif finish_reason == "content_filter":
            # Content filter with status 200 indicates that the model output was filtered
            # https://learn.microsoft.com/en-us/azure/ai-services/openai/concepts/content-filter
            return handle_bad_request_exception(
                response_text=open_ai_str_response, request=request_piece, error_code=200, is_content_filter=True
            )
        else:
            raise PyritException(message=f"Unknown finish_reason {finish_reason} from response: {response}")

        return construct_response_from_request(request=request_piece, response_text_pieces=[extracted_response])

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
