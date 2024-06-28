# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
import json

from openai import AsyncAzureOpenAI
from openai import BadRequestError
from openai.types.chat import ChatCompletion


from pyrit.auth.azure_auth import get_token_provider_from_default_azure_credential
from pyrit.common import default_values
from pyrit.exceptions import PyritException, EmptyResponseException
from pyrit.exceptions import handle_bad_request_exception, pyrit_target_retry
from pyrit.memory import MemoryInterface
from pyrit.models import ChatMessageListContent, PromptRequestResponse, PromptRequestPiece, DataTypeSerializer
from pyrit.models import data_serializer_factory, construct_response_from_request
from pyrit.prompt_target import PromptChatTarget

logger = logging.getLogger(__name__)

# Supported image formats for Azure OpenAI GPT-V,
# https://learn.microsoft.com/en-us/azure/ai-services/openai/concepts/use-your-image-data
AZURE_OPENAI_GPTV_SUPPORTED_IMAGE_FORMATS = [".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", "tif"]


class AzureOpenAIGPTVChatTarget(PromptChatTarget):
    """This class facilitates multimodal (image and text) input and text output generation"""

    API_KEY_ENVIRONMENT_VARIABLE: str = "AZURE_OPENAI_GPTV_CHAT_KEY"
    ENDPOINT_URI_ENVIRONMENT_VARIABLE: str = "AZURE_OPENAI_GPTV_CHAT_ENDPOINT"
    DEPLOYMENT_ENVIRONMENT_VARIABLE: str = "AZURE_OPENAI_GPTV_CHAT_DEPLOYMENT"
    ADDITIONAL_REQUEST_HEADERS: str = "AZURE_OPENAI_CHAT_ADDITIONAL_REQUEST_HEADERS"

    def __init__(
        self,
        *,
        deployment_name: str = None,
        endpoint: str = None,
        api_key: str = None,
        headers: str = None,
        use_aad_auth: bool = False,
        memory: MemoryInterface = None,
        api_version: str = "2024-02-01",
        max_tokens: int = 1024,
        temperature: float = 1.0,
        top_p: int = 1,
        frequency_penalty: float = 0.5,
        presence_penalty: float = 0.5,
    ) -> None:
        """
        Class that initializes an Azure Open AI GPTV chat target

        Args:
            deployment_name (str, optional): The name of the deployment. Defaults to the
                DEPLOYMENT_ENVIRONMENT_VARIABLE environment variable .
            endpoint (str, optional): The endpoint URL for the Azure OpenAI service.
                Defaults to the ENDPOINT_URI_ENVIRONMENT_VARIABLE environment variable.
            api_key (str, optional): The API key for accessing the Azure OpenAI service.
                Defaults to the API_KEY_ENVIRONMENT_VARIABLE environment variable.
            use_aad_auth (bool, optional): When set to True, user authentication is used
                instead of API Key. DefaultAzureCredential is taken for
                https://cognitiveservices.azure.com/.default. Please run `az login` locally
                to leverage user AuthN.
            memory (MemoryInterface, optional): An instance of the MemoryInterface class
                for storing conversation history. Defaults to None.
            api_version (str, optional): The version of the Azure OpenAI API. Defaults to
                "2024-02-01".
            headers (str, optional): Headers of the endpoint.
            max_tokens (int, optional): The maximum number of tokens to generate in the response.
                Defaults to 1024.
            temperature (float, optional): The temperature parameter for controlling the
                randomness of the response. Defaults to 1.0.
            top_p (int, optional): The top-p parameter for controlling the diversity of the
                response. Defaults to 1.
            frequency_penalty (float, optional): The frequency penalty parameter for penalizing
                frequently generated tokens. Defaults to 0.5.
            presence_penalty (float, optional): The presence penalty parameter for penalizing
                tokens that are already present in the conversation history. Defaults to 0.5.
        """
        PromptChatTarget.__init__(self, memory=memory)

        self._max_tokens = max_tokens
        self._temperature = temperature
        self._top_p = top_p
        self._frequency_penalty = frequency_penalty
        self._presence_penalty = presence_penalty

        self._deployment_name = default_values.get_required_value(
            env_var_name=self.DEPLOYMENT_ENVIRONMENT_VARIABLE, passed_value=deployment_name
        )
        endpoint = default_values.get_required_value(
            env_var_name=self.ENDPOINT_URI_ENVIRONMENT_VARIABLE, passed_value=endpoint
        )

        final_headers: dict = {}
        try:
            request_headers = default_values.get_required_value(
                env_var_name=self.ADDITIONAL_REQUEST_HEADERS, passed_value=headers
            )
            if isinstance(request_headers, str):
                try:
                    final_headers = json.loads(request_headers)
                except json.JSONDecodeError as e:
                    logger.error(f"Error decoding JSON: {e}")
        except ValueError:
            logger.info("No headers have been passed, setting empty default headers")

        if use_aad_auth:
            logger.info("Authenticating with DefaultAzureCredential() for Azure Cognitive Services")
            token_provider = get_token_provider_from_default_azure_credential()

            self._async_client = AsyncAzureOpenAI(
                azure_ad_token_provider=token_provider,
                api_version=api_version,
                azure_endpoint=endpoint,
                default_headers=final_headers,
            )
        else:
            api_key = default_values.get_required_value(
                env_var_name=self.API_KEY_ENVIRONMENT_VARIABLE, passed_value=api_key
            )

            self._async_client = AsyncAzureOpenAI(
                api_key=api_key, api_version=api_version, azure_endpoint=endpoint, default_headers=final_headers
            )

    async def send_prompt_async(self, *, prompt_request: PromptRequestResponse) -> PromptRequestResponse:
        """Asynchronously sends a prompt request and handles the response within a managed conversation context.

        Args:
            prompt_request (PromptRequestResponse): The prompt request response object.

        Returns:
            PromptRequestResponse: The updated conversation entry with the response from the prompt target.
        """
        self._validate_request(prompt_request=prompt_request)
        request: PromptRequestPiece = prompt_request.request_pieces[0]

        prompt_req_res_entries = self._memory.get_conversation(conversation_id=request.conversation_id)
        prompt_req_res_entries.append(prompt_request)

        logger.info(f"Sending the following prompt to the prompt target: {prompt_request}")

        messages = self._build_chat_messages(prompt_req_res_entries)
        try:
            resp_text = await self._complete_chat_async(
                messages=messages,
                top_p=self._top_p,
                temperature=self._temperature,
                frequency_penalty=self._frequency_penalty,
                presence_penalty=self._presence_penalty,
            )

            logger.info(f'Received the following response from the prompt target "{resp_text}"')

            response_entry = construct_response_from_request(request=request, response_text_pieces=[resp_text])
        except BadRequestError as bre:
            response_entry = handle_bad_request_exception(response_text=bre.message, request=request)

        return response_entry

    def _convert_local_image_to_data_url(self, image_path: str) -> str:
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
        if ext.lower() not in AZURE_OPENAI_GPTV_SUPPORTED_IMAGE_FORMATS:
            raise ValueError(
                f"Unsupported image format: {ext}. Supported formats are: {AZURE_OPENAI_GPTV_SUPPORTED_IMAGE_FORMATS}"
            )

        mime_type = DataTypeSerializer.get_mime_type(image_path)
        if not mime_type:
            mime_type = "application/octet-stream"

        image_serializer = data_serializer_factory(value=image_path, data_type="image_path", extension=ext)
        base64_encoded_data = image_serializer.read_data_base64()
        # Construct the data URL, as per Azure OpenAI GPTV local image format
        # https://learn.microsoft.com/en-us/azure/ai-services/openai/how-to/gpt-with-vision?tabs=rest%2Csystem-assigned%2Cresource#use-a-local-image
        return f"data:{mime_type};base64,{base64_encoded_data}"

    def _build_chat_messages(self, prompt_req_res_entries: list[PromptRequestResponse]) -> list[ChatMessageListContent]:
        """
        Builds chat messages based on prompt request response entries.

        Args:
            prompt_req_res_entries (list[PromptRequestResponse]): A list of PromptRequestResponse objects.

        Returns:
            list[ChatMessageListContent]: The list of constructed chat messages.
        """
        chat_messages: list[ChatMessageListContent] = []
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
                    data_base64_encoded_url = self._convert_local_image_to_data_url(
                        prompt_request_piece.converted_value
                    )
                    entry = {"type": "image_url", "image_url": data_base64_encoded_url}
                    content.append(entry)
                else:
                    raise ValueError(
                        f"Multimodal data type {prompt_request_piece.converted_value_data_type} is not yet supported."
                    )

            if not role:
                raise ValueError("No role could be determined from the prompt request pieces.")

            chat_message = ChatMessageListContent(role=role, content=content)
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
    async def _complete_chat_async(
        self,
        messages: list[ChatMessageListContent],
        max_tokens: int = 1024,
        temperature: float = 1.0,
        top_p: int = 1,
        frequency_penalty: float = 0.5,
        presence_penalty: float = 0.5,
    ) -> str:
        """
        Completes asynchronous chat request.

        Sends a chat message to the OpenAI chat model and retrieves the generated response.

        Args:
            messages (list[ChatMessageListContent]): The chat message objects containing the role and content.
            max_tokens (int, optional): The maximum number of tokens to generate.
                Defaults to 1024.
            temperature (float, optional): Controls randomness in the response generation.
                Defaults to 1.0.
            top_p (int, optional): Controls diversity of the response generation.
                Defaults to 1.
            frequency_penalty (float, optional): Controls the frequency of generating the same lines of text.
                Defaults to 0.5.
            presence_penalty (float, optional): Controls the likelihood to talk about new topics.
                Defaults to 0.5.

        Returns:
            str: The generated response message.
        """

        response: ChatCompletion = await self._async_client.chat.completions.create(
            model=self._deployment_name,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            n=1,
            stream=False,
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
        # As of April 26, 2024, the Azure OpenAI GPT-V model only accepts text and image as input.
        if len(prompt_request.request_pieces) > 2:
            raise ValueError("This target only supports a two prompt request pieces text and image_path.")

        converted_prompt_data_types = [
            request_piece.converted_value_data_type for request_piece in prompt_request.request_pieces
        ]
        for prompt_data_type in converted_prompt_data_types:
            if prompt_data_type not in ["text", "image_path"]:
                raise ValueError("This target only supports text and image_path.")
