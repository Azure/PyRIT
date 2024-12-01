# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
from typing import MutableSequence, Optional

from openai import BadRequestError, NotGiven, NOT_GIVEN
from openai.types.chat import ChatCompletion


from pyrit.exceptions import PyritException, EmptyResponseException
from pyrit.exceptions import handle_bad_request_exception, pyrit_target_retry
from pyrit.models import ChatMessageListDictContent, PromptRequestResponse, PromptRequestPiece, DataTypeSerializer
from pyrit.models import data_serializer_factory, construct_response_from_request
from pyrit.prompt_target import OpenAITarget, limit_requests_per_minute


logger = logging.getLogger(__name__)

# Supported image formats for Azure OpenAI GPT-4o,
# https://learn.microsoft.com/en-us/azure/ai-services/openai/concepts/use-your-image-data
AZURE_OPENAI_GPT4O_SUPPORTED_IMAGE_FORMATS = [".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", "tif"]


class OpenAIChatTarget(OpenAITarget):
    """
    This class facilitates multimodal (image and text) input and text output generation

    This works with GPT3.5, GPT4, GPT4o, GPT-V, and other compatible models
    """

    def __init__(
        self,
        max_completion_tokens: Optional[int] | NotGiven = NOT_GIVEN,
        max_tokens: Optional[int] | NotGiven = NOT_GIVEN,
        temperature: float = 1.0,
        top_p: float = 1.0,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        seed: Optional[int] = None,
        *args,
        **kwargs,
    ):
        """
        Args:
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
                randomness of the response. Defaults to 1.0.
            top_p (float, Optional): The top-p parameter for controlling the diversity of the
                response. Defaults to 1.0.
            frequency_penalty (float, Optional): The frequency penalty parameter for penalizing
                frequently generated tokens. Defaults to 0.
            presence_penalty (float, Optional): The presence penalty parameter for penalizing
                tokens that are already present in the conversation history. Defaults to 0.
            seed (int, Optional): If specified, openAI will make a best effort to sample deterministically,
                such that repeated requests with the same seed and parameters should return the same result.
        """
        super().__init__(*args, **kwargs)

        if max_completion_tokens is not NOT_GIVEN and max_tokens is not NOT_GIVEN:
            raise ValueError("Cannot provide both max_tokens and max_completion_tokens.")

        self._max_completion_tokens = max_completion_tokens
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._top_p = top_p
        self._frequency_penalty = frequency_penalty
        self._presence_penalty = presence_penalty
        self._seed = seed

    def _set_azure_openai_env_configuration_vars(self) -> None:
        self.deployment_environment_variable = "AZURE_OPENAI_CHAT_DEPLOYMENT"
        self.endpoint_uri_environment_variable = "AZURE_OPENAI_CHAT_ENDPOINT"
        self.api_key_environment_variable = "AZURE_OPENAI_CHAT_KEY"

    @limit_requests_per_minute
    async def send_prompt_async(self, *, prompt_request: PromptRequestResponse) -> PromptRequestResponse:
        """Asynchronously sends a prompt request and handles the response within a managed conversation context.

        Args:
            prompt_request (PromptRequestResponse): The prompt request response object.

        Returns:
            PromptRequestResponse: The updated conversation entry with the response from the prompt target.
        """
        self._validate_request(prompt_request=prompt_request)
        request_piece: PromptRequestPiece = prompt_request.request_pieces[0]

        prompt_req_res_entries = self._memory.get_conversation(conversation_id=request_piece.conversation_id)
        prompt_req_res_entries.append(prompt_request)

        logger.info(f"Sending the following prompt to the prompt target: {prompt_request}")

        messages = await self._build_chat_messages(prompt_req_res_entries)
        try:
            resp_text = await self._complete_chat_async(messages=messages)

            logger.info(f'Received the following response from the prompt target "{resp_text}"')

            response_entry = construct_response_from_request(request=request_piece, response_text_pieces=[resp_text])
        except BadRequestError as bre:
            response_entry = handle_bad_request_exception(response_text=bre.message, request=request_piece)

        return response_entry

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

        image_serializer = data_serializer_factory(value=image_path, data_type="image_path", extension=ext)
        base64_encoded_data = await image_serializer.read_data_base64()
        # Azure OpenAI GPT-4o documentation doesn't specify the local image upload format for API.
        # GPT-4o image upload format is determined using "view code" functionality in Azure OpenAI deployments
        # The image upload format is same as GPT-4 Turbo.
        # Construct the data URL, as per Azure OpenAI GPT-4 Turbo local image format
        # https://learn.microsoft.com/en-us/azure/ai-services/openai/how-to/gpt-with-vision?tabs=rest%2Csystem-assigned%2Cresource#call-the-chat-completion-apis
        return f"data:{mime_type};base64,{base64_encoded_data}"

    async def _build_chat_messages(
        self, prompt_req_res_entries: MutableSequence[PromptRequestResponse]
    ) -> list[ChatMessageListDictContent]:
        """
        Builds chat messages based on prompt request response entries.

        Args:
            prompt_req_res_entries (list[PromptRequestResponse]): A list of PromptRequestResponse objects.

        Returns:
            list[ChatMessageListDictContent]: The list of constructed chat messages.
        """
        chat_messages: list[ChatMessageListDictContent] = []
        for prompt_req_resp_entry in prompt_req_res_entries:
            prompt_request_pieces = prompt_req_resp_entry.request_pieces

            content = []
            role = None
            for prompt_request_piece in prompt_request_pieces:
                role = prompt_request_piece.role
                if prompt_request_piece.converted_value_data_type == "text":
                    entry = {"type": "text", "text": prompt_request_piece.converted_value}
                    content.append(entry)
                elif prompt_request_piece.converted_value_data_type == "image_path":
                    data_base64_encoded_url = await self._convert_local_image_to_data_url(
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
            chat_messages.append(chat_message)
        return chat_messages

    def _parse_chat_completion(self, response):
        """
        Parses chat message to get response

        Args:
            response (ChatMessage): The chat messages object containing the generated response message

        Returns:
            str: The generated response message
        """
        response_message = response.choices[0].message.content
        return response_message

    @pyrit_target_retry
    async def _complete_chat_async(self, messages: list[ChatMessageListDictContent]) -> str:
        """
        Completes asynchronous chat request.

        Sends a chat message to the OpenAI chat model and retrieves the generated response.

        Args:
            messages (list[ChatMessageListDictContent]): The chat message objects containing the role and content.

        Returns:
            str: The generated response message.
        """

        response: ChatCompletion = await self._async_client.chat.completions.create(
            model=self._deployment_name,
            max_completion_tokens=self._max_completion_tokens,
            max_tokens=self._max_tokens,
            temperature=self._temperature,
            top_p=self._top_p,
            frequency_penalty=self._frequency_penalty,
            presence_penalty=self._presence_penalty,
            n=1,
            stream=False,
            seed=self._seed,
            messages=[{"role": msg.role, "content": msg.content} for msg in messages],  # type: ignore
        )
        finish_reason = response.choices[0].finish_reason
        extracted_response: str = ""
        # finish_reason="stop" means API returned complete message and
        # "length" means API returned incomplete message due to max_tokens limit.
        if finish_reason in ["stop", "length"]:
            extracted_response = self._parse_chat_completion(response)
            # Handle empty response
            if not extracted_response:
                logger.log(logging.ERROR, "The chat returned an empty response.")
                raise EmptyResponseException(message="The chat returned an empty response.")
        else:
            raise PyritException(message=f"Unknown finish_reason {finish_reason}")

        return extracted_response

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
