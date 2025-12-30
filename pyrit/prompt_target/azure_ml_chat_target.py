# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
from typing import Optional

from httpx import HTTPStatusError

from pyrit.message_normalizer import MessageNop, MessageListNormalizer, ChatMessageNormalizer
from pyrit.common import default_values, net_utility
from pyrit.exceptions import (
    EmptyResponseException,
    RateLimitException,
    handle_bad_request_exception,
    pyrit_target_retry,
)
from pyrit.models import (
    ChatMessage,
    Message,
    construct_response_from_request,
)
from pyrit.prompt_target import PromptChatTarget, limit_requests_per_minute
from pyrit.prompt_target.common.utils import validate_temperature, validate_top_p

logger = logging.getLogger(__name__)


class AzureMLChatTarget(PromptChatTarget):
    """
    A prompt target for Azure Machine Learning chat endpoints.

    This class works with most chat completion Instruct models deployed on Azure AI Machine Learning
    Studio endpoints (including but not limited to: mistralai-Mixtral-8x7B-Instruct-v01,
    mistralai-Mistral-7B-Instruct-v01, Phi-3.5-MoE-instruct, Phi-3-mini-4k-instruct,
    Llama-3.2-3B-Instruct, and Meta-Llama-3.1-8B-Instruct).

    Please create or adjust environment variables (endpoint and key) as needed for the model you are using.
    """

    endpoint_uri_environment_variable: str = "AZURE_ML_MANAGED_ENDPOINT"
    api_key_environment_variable: str = "AZURE_ML_KEY"

    def __init__(
        self,
        *,
        endpoint: Optional[str] = None,
        api_key: Optional[str] = None,
        message_normalizer: MessageListNormalizer = MessageNop(),
        max_new_tokens: int = 400,
        temperature: float = 1.0,
        top_p: float = 1.0,
        repetition_penalty: float = 1.0,
        max_requests_per_minute: Optional[int] = None,
        **param_kwargs,
    ) -> None:
        """
        Initialize an instance of the AzureMLChatTarget class.

        Args:
            endpoint (str, Optional): The endpoint URL for the deployed Azure ML model.
                Defaults to the value of the AZURE_ML_MANAGED_ENDPOINT environment variable.
            api_key (str, Optional): The API key for accessing the Azure ML endpoint.
                Defaults to the value of the `AZURE_ML_KEY` environment variable.
            message_normalizer (MessageListNormalizer, Optional): The message normalizer.
                For models that do not allow system prompts such as mistralai-Mixtral-8x7B-Instruct-v01,
                GenericSystemSquashNormalizer() can be passed in. Defaults to MessageNop(), which does not
                alter the chat messages.
            max_new_tokens (int, Optional): The maximum number of tokens to generate in the response.
                Defaults to 400.
            temperature (float, Optional): The temperature for generating diverse responses. 1.0 is most random,
                0.0 is least random. Defaults to 1.0.
            top_p (float, Optional): The top-p value for generating diverse responses. It represents
                the cumulative probability of the top tokens to keep. Defaults to 1.0.
            repetition_penalty (float, Optional): The repetition penalty for generating diverse responses.
                1.0 means no penalty with a greater value (up to 2.0) meaning more penalty for repeating tokens.
                Defaults to 1.2.
            max_requests_per_minute (int, Optional): Number of requests the target can handle per
                minute before hitting a rate limit. The number of requests sent to the target
                will be capped at the value provided.
            **param_kwargs: Additional parameters to pass to the model for generating responses. Example
                parameters can be found here: https://huggingface.co/docs/api-inference/tasks/text-generation.
                Note that the link above may not be comprehensive, and specific acceptable parameters may be
                model-dependent. If a model does not accept a certain parameter that is passed in, it will be skipped
                without throwing an error.
        """
        endpoint_value = default_values.get_required_value(
            env_var_name=self.endpoint_uri_environment_variable, passed_value=endpoint
        )
        PromptChatTarget.__init__(self, max_requests_per_minute=max_requests_per_minute, endpoint=endpoint_value)

        self._initialize_vars(endpoint=endpoint, api_key=api_key)

        validate_temperature(temperature)
        validate_top_p(top_p)

        self.message_normalizer = message_normalizer
        self._max_new_tokens = max_new_tokens
        self._temperature = temperature
        self._top_p = top_p
        self._repetition_penalty = repetition_penalty
        self._extra_parameters = param_kwargs

    def _set_env_configuration_vars(
        self,
        endpoint_uri_environment_variable: Optional[str] = None,
        api_key_environment_variable: Optional[str] = None,
    ) -> None:
        """
        Set the environment configuration variable names from which to pull the endpoint uri and the api key
        to access the deployed Azure ML model. Use this function to set the environment variable names to
        however they are named in the .env file and pull the corresponding endpoint uri and api key.
        This is the recommended way to pass in a uri and key to access the model endpoint.
        Defaults to "AZURE_ML_MANAGED_ENDPOINT" and "AZURE_ML_KEY".

        Args:
            endpoint_uri_environment_variable (str, optional): The environment variable name for the endpoint uri.
            api_key_environment_variable (str, optional): The environment variable name for the api key.
        """
        self.endpoint_uri_environment_variable = endpoint_uri_environment_variable or "AZURE_ML_MANAGED_ENDPOINT"
        self.api_key_environment_variable = api_key_environment_variable or "AZURE_ML_KEY"
        self._initialize_vars()

    def _initialize_vars(self, endpoint: Optional[str] = None, api_key: Optional[str] = None) -> None:
        """
        Set the endpoint and key for accessing the Azure ML model. Use this function to manually
        pass in your own endpoint uri and api key. Defaults to the values in the .env file for the variables
        stored in self.endpoint_uri_environment_variable and self.api_key_environment_variable (which default to
        "AZURE_ML_MANAGED_ENDPOINT" and "AZURE_ML_KEY" respectively). It is recommended to set these variables
        in the .env file and call _set_env_configuration_vars rather than passing the uri and key directly to
        this function or the target constructor.

        Args:
            endpoint (str, optional): The endpoint uri for the deployed Azure ML model.
            api_key (str, optional): The API key for accessing the Azure ML endpoint.
        """
        self._endpoint = default_values.get_required_value(
            env_var_name=self.endpoint_uri_environment_variable, passed_value=endpoint
        )
        self._api_key = default_values.get_required_value(
            env_var_name=self.api_key_environment_variable, passed_value=api_key
        )

    def _set_model_parameters(
        self,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        repetition_penalty: Optional[float] = None,
        **param_kwargs,
    ) -> None:
        """
        Set the model parameters for generating responses, offering the option to add additional ones not
        explicitly listed.

        Args:
            max_new_tokens: Maximum number of new tokens to generate.
            temperature: Sampling temperature for response generation.
            top_p: Nucleus sampling parameter.
            repetition_penalty: Penalty for repeating tokens.
            **param_kwargs: Additional model parameters.
        """
        self._max_new_tokens = max_new_tokens or self._max_new_tokens
        self._temperature = temperature or self._temperature
        self._top_p = top_p or self._top_p
        self._repetition_penalty = repetition_penalty or self._repetition_penalty
        # Set any other parameters via additional keyword arguments
        self._extra_parameters = param_kwargs

    @limit_requests_per_minute
    async def send_prompt_async(self, *, message: Message) -> list[Message]:
        """
        Asynchronously send a message to the Azure ML chat target.

        Args:
            message (Message): The message object containing the prompt to send.

        Returns:
            list[Message]: A list containing the response from the prompt target.

        Raises:
            EmptyResponseException: If the response from the chat is empty.
            RateLimitException: If the target rate limit is exceeded.
            HTTPStatusError: For any other HTTP errors during the process.
        """
        self._validate_request(message=message)
        request = message.message_pieces[0]

        # Get chat messages from memory and append the current message
        messages = list(self._memory.get_chat_messages_with_conversation_id(conversation_id=request.conversation_id))
        # Convert the current message piece to ChatMessage format
        current_chat_message = ChatMessage(role=request.role, content=request.converted_value)  # type: ignore
        messages.append(current_chat_message)

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

        logger.info("Received the following response from the prompt target" + f"{response_entry.get_value()}")
        return [response_entry]

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
            EmptyResponseException: If the response from the chat is empty.
            Exception: For any other errors during the process.

        Returns:
            str: The generated response message.
        """
        headers = self._get_headers()
        payload = await self._construct_http_body_async(messages)

        response = await net_utility.make_request_and_raise_if_error_async(
            endpoint_uri=self._endpoint, method="POST", request_body=payload, headers=headers
        )

        try:
            return response.json()["output"]
        except Exception as e:
            if response.json() == {}:
                raise EmptyResponseException(message="The chat returned an empty response.")
            raise e(
                f"Exception obtaining response from the target. Returned response: {response.json()}. "
                + f"Exception: {str(e)}"  # type: ignore
            )

    async def _construct_http_body_async(
        self,
        messages: list[ChatMessage],
    ) -> dict:
        """
        Construct the HTTP request body for the AML online endpoint.

        Args:
            messages: List of chat messages to include in the request body.

        Returns:
            dict: The constructed HTTP request body.
        """
        # Convert ChatMessages to Messages for the normalizer
        message_list = [
            Message.from_prompt(prompt=msg.content or "", role=msg.role)
            for msg in messages
        ]
        # Normalize messages (e.g., squash system message into first user message)
        normalized_messages = await self.message_normalizer.normalize_async(message_list)
        # Convert back to dict format for the API
        messages_dict = [
            {"role": m.role, "content": m.get_value()} for m in normalized_messages
        ]

        # parameters include additional ones passed in through **kwargs. Those not accepted by the model will
        # be ignored.
        data = {
            "input_data": {
                "input_string": messages_dict,
                "parameters": {
                    "max_new_tokens": self._max_new_tokens,
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
        """
        Headers for accessing inference endpoint deployed in AML.

        Returns:
            headers(dict): contains bearer token as AML key and content-type: JSON
        """
        headers: dict = {
            "Content-Type": "application/json",
            "Authorization": ("Bearer " + self._api_key),
        }

        return headers

    def _validate_request(self, *, message: Message) -> None:
        n_pieces = len(message.message_pieces)
        if n_pieces != 1:
            raise ValueError(f"This target only supports a single message piece. Received: {n_pieces} pieces.")

        piece_type = message.message_pieces[0].converted_value_data_type
        if piece_type != "text":
            raise ValueError(f"This target only supports text prompt input. Received: {piece_type}.")

    def is_json_response_supported(self) -> bool:
        """
        Check if the target supports JSON as a response format.

        Returns:
            bool: True if JSON response is supported, False otherwise.
        """
        return False
