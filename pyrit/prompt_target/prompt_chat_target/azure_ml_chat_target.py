# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
from httpx import HTTPStatusError
from typing import get_args, Literal, Optional, Tuple

from pyrit.chat_message_normalizer import ChatMessageNop, ChatMessageNormalizer
from pyrit.common import default_values, net_utility
from pyrit.exceptions import EmptyResponseException, RateLimitException
from pyrit.exceptions import handle_bad_request_exception, pyrit_target_retry
from pyrit.memory import MemoryInterface
from pyrit.models import ChatMessage, PromptRequestResponse
from pyrit.models import construct_response_from_request
from pyrit.prompt_target import PromptChatTarget, limit_requests_per_minute

logger = logging.getLogger(__name__)

"""
Azure AI Machine Learning Studio model names can be found by clicking into the Endpoint and clicking the link
under the 'Model ID' on the right hand side. Alternatively, names can be found in the Model Catalog,
which can be found when logging in here: https://ml.azure.com/home, choosing a workspace, and navigating
to 'Model Catalog.'. The below will be modified based on models made available through dedicated endpoints.
"""
AzureMLName = Literal[
    "mistralai-Mixtral-8x7B-Instruct-v01", "mistralai-Mistral-7B-Instruct-v01", "Phi-3-mini-4k-instruct"
]


def get_env_variable_from_model_name(model: AzureMLName | str) -> str:
    """
    Returns the environment variable name based on the model name. Note: make sure that the environment variables
    in .env file are named in the format: {AZUREML_MODEL_NAME}_MANAGED_ENDPOINT and {AZUREML_MODEL_NAME}_KEY
    """
    return model.upper().replace("-", "_")


class AzureMLChatTarget(PromptChatTarget):

    def __init__(
        self,
        *,
        model: AzureMLName | str = "mistralai-Mixtral-8x7B-Instruct-v01",
        endpoint_uri_and_key: Tuple[str, str] = None,
        chat_message_normalizer: ChatMessageNormalizer = ChatMessageNop(),
        memory: MemoryInterface = None,
        max_tokens: int = 400,
        temperature: float = 1.0,
        top_p: float = 1.0,
        repetition_penalty: float = 1.2,
        max_requests_per_minute: Optional[int] = None,
        **param_kwargs,
    ) -> None:
        """
        Initializes an instance of the AzureMLChatTarget class.

        Args:
            model (AzureMLName, optional): The AzureML model name to use for the chat target.
                Environment variables will be populated based on the model name provided and
                AML_NAME_TO_ENVIRONMENT_VARIABLE map. Defaults to None.
            endpoint_uri_and_key (Tuple[str, str], optional): Tuple containing the endpoint URI and key.
                Note: Only one of model or endpoint_uri_and_key should be provided. Defaults to None.
            chat_message_normalizer (ChatMessageNormalizer, optional): The chat message normalizer.
                For models that do not allow system prompts such as mistralai-Mixtral-8x7B-Instruct-v01,
                GenericSystemSquash() can be passed in. Defaults to ChatMessageNop(), which does not
                alter the chat messages.
            memory (MemoryInterface, optional): The memory interface.
                Defaults to None.
            max_tokens (int, optional): The maximum number of tokens to generate in the response.
                Defaults to 400.
            temperature (float, optional): The temperature for generating diverse responses. 1.0 is most random,
                0.0 is least random. Defaults to 1.0.
            top_p (float, optional): The top-p value for generating diverse responses. It represents
                the cumulative probability of the top tokens to keep. Defaults to 1.0.
            repetition_penalty (float, optional): The repetition penalty for generating diverse responses.
                1.0 means no penalty with a greater value (up to 2.0) meaning more penalty for repeating tokens.
                Defaults to 1.2.
            max_requests_per_minute (int, optional): Number of requests the target can handle per
                minute before hitting a rate limit. The number of requests sent to the target
                will be capped at the value provided.
            **param_kwargs: Additional parameters to pass to the model for generating responses. Example
                parameters can be found here: https://huggingface.co/docs/api-inference/tasks/text-generation
        """
        PromptChatTarget.__init__(self, memory=memory, max_requests_per_minute=max_requests_per_minute)

        if model not in get_args(AzureMLName):
            logger.info(
                f"Model name {model} not in the defaults provided here: {get_args(AzureMLName)}. \
                        Please make sure the model name is correct."
            )
        self._model = model
        self.endpoint_uri: str = default_values.get_required_value(
            env_var_name=f"{get_env_variable_from_model_name(model)}_MANAGED_ENDPOINT",
            passed_value=None if not endpoint_uri_and_key else endpoint_uri_and_key[0],
        )
        self.api_key: str = default_values.get_required_value(
            env_var_name=f"{get_env_variable_from_model_name(model)}_KEY",
            passed_value=None if not endpoint_uri_and_key else endpoint_uri_and_key[1],
        )
        self.chat_message_normalizer = chat_message_normalizer
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._top_p = top_p
        self._repetition_penalty = repetition_penalty
        self._extra_parameters = param_kwargs

    def _set_model_parameters(
        self,
        max_tokens: int = None,
        temperature: float = None,
        top_p: float = None,
        repetition_penalty: float = None,
        **param_kwargs,
    ) -> None:
        """Sets the model parameters for generating responses."""
        self._max_tokens = max_tokens or self._max_tokens
        self._temperature = temperature or self._temperature
        self._top_p = top_p or self._top_p
        self._repetition_penalty = repetition_penalty or self._repetition_penalty
        # Set any other parameters via additional keyword arguments
        self._extra_parameters = param_kwargs

    @limit_requests_per_minute
    async def send_prompt_async(self, *, prompt_request: PromptRequestResponse) -> PromptRequestResponse:

        self._validate_request(prompt_request=prompt_request)
        request = prompt_request.request_pieces[0]

        messages = self._memory.get_chat_messages_with_conversation_id(conversation_id=request.conversation_id)

        messages.append(request.to_chat_message())

        logger.info(f"Sending the following prompt to the prompt target: {request}")

        try:
            resp_text = await self._complete_chat_async(
                messages=messages,
            )

            if not resp_text:
                raise EmptyResponseException(message="The chat returned an empty response.")

            response_entry = construct_response_from_request(request=request, response_text_pieces=[resp_text])
        except HTTPStatusError as hse:
            if hse.response.status_code == 400:
                # Handle Bad Request
                response_entry = handle_bad_request_exception(response_text=hse.response.text, request=request)
            elif hse.response.status_code == 429:
                raise RateLimitException()
            else:
                raise hse

        logger.info(
            "Received the following response from the prompt target"
            + f"{response_entry.request_pieces[0].converted_value}"
        )
        return response_entry

    @pyrit_target_retry
    async def _complete_chat_async(
        self,
        messages: list[ChatMessage],
    ) -> str:
        """
        Completes a chat interaction by generating a response to the given input prompt.

        This is a synchronous wrapper for the asynchronous _generate_and_extract_response method.

        Args:
            messages (list[ChatMessage]): The chat messages objects containing the role and content.

        Raises:
            Exception: For any errors during the process.

        Returns:
            str: The generated response message.
        """
        headers = self._get_headers()
        payload = self._construct_http_body(messages)

        response = await net_utility.make_request_and_raise_if_error_async(
            endpoint_uri=self.endpoint_uri, method="POST", request_body=payload, headers=headers
        )

        return response.json()["output"]

    def _construct_http_body(
        self,
        messages: list[ChatMessage],
    ) -> dict:
        """Constructs the HTTP request body for the AML online endpoint."""

        squashed_messages = self.chat_message_normalizer.normalize(messages)
        messages_dict = [message.model_dump() for message in squashed_messages]

        # parameters include additional ones passed in through **kwargs
        data = {
            "input_data": {
                "input_string": messages_dict,
                "parameters": {
                    "max_new_tokens": self._max_tokens,
                    "temperature": self._temperature,
                    "top_p": self._top_p,
                    "stop": ["</s>"],
                    "stop_sequences": ["</s>"],
                    "return_full_text": False,
                    "repetition_penalty": self._repetition_penalty,
                }
                | self._extra_parameters,
            }
        }

        return data

    def _get_headers(self) -> dict:
        """Headers for accessing inference endpoint deployed in AML.
        Returns:
            headers(dict): contains bearer token as AML key and content-type: JSON
        """

        headers: dict = {
            "Content-Type": "application/json",
            "Authorization": ("Bearer " + self.api_key),
        }

        return headers

    def _validate_request(self, *, prompt_request: PromptRequestResponse) -> None:
        if len(prompt_request.request_pieces) != 1:
            raise ValueError("This target only supports a single prompt request piece.")

        if prompt_request.request_pieces[0].converted_value_data_type != "text":
            raise ValueError("This target only supports text prompt input.")
