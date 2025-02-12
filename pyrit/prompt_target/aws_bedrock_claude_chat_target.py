import asyncio
import logging
import json
import boto3
from typing import MutableSequence, Optional

from botocore.exceptions import ClientError

from pyrit.chat_message_normalizer import ChatMessageNop, ChatMessageNormalizer
from pyrit.common import net_utility
from pyrit.models import ChatMessage, PromptRequestPiece, PromptRequestResponse, construct_response_from_request, ChatMessageListDictContent
from pyrit.prompt_target import PromptChatTarget, limit_requests_per_minute

logger = logging.getLogger(__name__)

class AWSBedrockClaudeChatTarget(PromptChatTarget):
    """
    This class initializes an AWS Bedrock target for any of the Anthropic Claude models.
    Local AWS credentials (typically stored in ~/.aws) are used for authentication.
    See the following for more information: https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-anthropic-claude-messages.html

    Parameters:
        model_id (str): The model ID for target claude model
        max_tokens (int): maximum number of tokens to generate
        temperature (float, optional): The amount of randomness injected into the response.
        top_p (float, optional): Use nucleus sampling
        top_k (int, optional): Only sample from the top K options for each subsequent token
        enable_ssl_verification (bool, optional): whether or not to perform SSL certificate verification
    """
    def __init__(
        self,
        *,
        model_id: str,
        max_tokens: int,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        enable_ssl_verification: bool = True,
        chat_message_normalizer: ChatMessageNormalizer = ChatMessageNop(),
        max_requests_per_minute: Optional[int] = None,
    ) -> None:
        super().__init__(max_requests_per_minute=max_requests_per_minute)

        self._model_id = model_id
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._top_p = top_p
        self._top_k = top_k
        self._enable_ssl_verification = enable_ssl_verification
        self.chat_message_normalizer = chat_message_normalizer

        self._system_prompt = ''

    @limit_requests_per_minute
    async def send_prompt_async(self, *, prompt_request: PromptRequestResponse) -> PromptRequestResponse:
        
        self._validate_request(prompt_request=prompt_request)
        request_piece = prompt_request.request_pieces[0]

        prompt_req_res_entries = self._memory.get_conversation(conversation_id=request_piece.conversation_id)
        prompt_req_res_entries.append(prompt_request)

        logger.info(f"Sending the following prompt to the prompt target: {prompt_request}")

        messages = self._build_chat_messages(prompt_req_res_entries)

        response = await self._complete_chat_async(messages=messages)

        response_entry = construct_response_from_request(request=request_piece, response_text_pieces=[response])

        return response_entry

    def _validate_request(self, *, prompt_request: PromptRequestResponse) -> None:
        if len(prompt_request.request_pieces) != 1:
            raise ValueError("This target only supports a single prompt request piece.")

        if prompt_request.request_pieces[0].converted_value_data_type != "text":
            raise ValueError("This target only supports text prompt input.")

    async def _complete_chat_async(self, messages: list[ChatMessageListDictContent]) -> str:
        brt = boto3.client(service_name="bedrock-runtime", region_name='us-east-1', enable_ssl_verification=self._enable_ssl_verification)

        native_request = self._construct_request_body(messages)

        request = json.dumps(native_request)

        try:
            response = await asyncio.to_thread(brt.invoke_model, modelId=self._model_id, body=request)
        except (ClientError, Exception) as e:
            raise ValueError(f"ERROR: Can't invoke '{self._model_id}'. Reason: {e}")

        model_response = json.loads(response["body"].read())

        answer = model_response["content"][0]["text"]

        logger.info(f'Received the following response from the prompt target "{answer}"')
        return answer

    def _build_chat_messages(self, prompt_req_res_entries: MutableSequence[PromptRequestResponse]
    ) -> list[ChatMessageListDictContent]:
        chat_messages: list[ChatMessageListDictContent] = []
        for prompt_req_resp_entry in prompt_req_res_entries:
            prompt_request_pieces = prompt_req_resp_entry.request_pieces

            content = []
            role = None
            for prompt_request_piece in prompt_request_pieces:
                role = prompt_request_piece.role
                if role == "system":
                    # Bedrock doesn't allow a message with role==system, but it does let you specify system role in a param
                    self._system_prompt = prompt_request_piece.converted_value
                elif prompt_request_piece.converted_value_data_type == "text":
                    entry = {"type": "text", "text": prompt_request_piece.converted_value}
                    content.append(entry)
                else:
                    raise ValueError(
                        f"Multimodal data type {prompt_request_piece.converted_value_data_type} is not yet supported."
                    )

            if not role:
                raise ValueError("No role could be determined from the prompt request pieces.")

            chat_message = ChatMessageListDictContent(role=role, content=content)
            chat_messages.append(chat_message)
        return chat_messages

    def _construct_request_body(self, messages_list: list[ChatMessageListDictContent]) -> dict:
        content = []

        for message in messages_list:
            if message.role != "system":
                entry = {"role": message.role, "content": message.content}
                content.append(entry)

        data = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": self._max_tokens,
            "system": self._system_prompt,
            "messages": content
        }

        if self._temperature:
            data['temperature'] = self._temperature
        if self._top_p:
            data['top_p'] = self._top_p
        if self._top_k:
            data['top_k'] = self._top_k

        return data

    def is_json_response_supported(self) -> bool:
        """Indicates that this target supports JSON response format."""
        return True
