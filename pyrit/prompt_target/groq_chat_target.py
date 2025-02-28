# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging

from openai import AsyncOpenAI
from openai.types.chat import ChatCompletion

from pyrit.common import default_values
from pyrit.exceptions import EmptyResponseException, PyritException, pyrit_target_retry
from pyrit.models import ChatMessageListDictContent
from pyrit.prompt_target.openai.openai_chat_target import OpenAIChatTarget

logger = logging.getLogger(__name__)


class GroqChatTarget(OpenAIChatTarget):
    """
    A chat target for interacting with Groq's OpenAI-compatible API.

    This class extends `OpenAIChatTarget` and ensures compatibility with Groq's API,
    which requires `msg.content` to be a string instead of a list of dictionaries.

    Attributes:
        API_KEY_ENVIRONMENT_VARIABLE (str): The environment variable for the Groq API key.
        MODEL_NAME_ENVIRONMENT_VARIABLE (str): The environment variable for the Groq model name.
        GROQ_API_BASE_URL (str): The fixed API base URL for Groq.
    """

    API_KEY_ENVIRONMENT_VARIABLE = "GROQ_API_KEY"
    MODEL_NAME_ENVIRONMENT_VARIABLE = "GROQ_MODEL_NAME"
    GROQ_API_BASE_URL = "https://api.groq.com/openai/v1/"

    def __init__(self, *, model_name: str = None, api_key: str = None, max_requests_per_minute: int = None, **kwargs):
        """
        Initializes GroqChatTarget with the correct API settings.

        Args:
            model_name (str, optional): The model to use. Defaults to `GROQ_MODEL_NAME` env variable.
            api_key (str, optional): The API key for authentication. Defaults to `GROQ_API_KEY` env variable.
            max_requests_per_minute (int, optional): Rate limit for requests.
        """

        kwargs.pop("endpoint", None)
        kwargs.pop("deployment_name", None)

        super().__init__(
            deployment_name=model_name,
            endpoint=self.GROQ_API_BASE_URL,
            api_key=api_key,
            is_azure_target=False,
            max_requests_per_minute=max_requests_per_minute,
            **kwargs,
        )

    def _initialize_non_azure_vars(self, deployment_name: str, endpoint: str, api_key: str):
        """
        Initializes variables to communicate with the (non-Azure) OpenAI API, in this case Groq.

        Args:
            deployment_name (str): The model name.
            endpoint (str): The API base URL.
            api_key (str): The API key.

        Raises:
            ValueError: If _deployment_name or _api_key is missing.
        """
        self._api_key = default_values.get_required_value(
            env_var_name=self.API_KEY_ENVIRONMENT_VARIABLE, passed_value=api_key
        )
        if not self._api_key:
            raise ValueError("API key for Groq is missing. Ensure GROQ_API_KEY is set in the environment.")

        self._deployment_name = default_values.get_required_value(
            env_var_name=self.MODEL_NAME_ENVIRONMENT_VARIABLE, passed_value=deployment_name
        )
        if not self._deployment_name:
            raise ValueError("Model name for Groq is missing. Ensure GROQ_MODEL_NAME is set in the environment.")

        # Ignoring mypy type error. The OpenAI client and Azure OpenAI client have the same private base class
        self._async_client = AsyncOpenAI(  # type: ignore
            api_key=self._api_key, default_headers=self._headers, base_url=endpoint
        )

    @pyrit_target_retry
    async def _complete_chat_async(self, messages: list[ChatMessageListDictContent], is_json_response: bool) -> str:
        """
        Completes asynchronous chat request.

        Sends a chat message to the OpenAI chat model and retrieves the generated response.
        This method modifies the request structure to ensure compatibility with Groq,
        which requires `msg.content` as a string instead of a list of dictionaries.
        msg.content -> msg.content[0].get("text")

        Args:
            messages (list[ChatMessageListDictContent]): The chat message objects containing the role and content.
            is_json_response (bool): Boolean indicating if the response should be in JSON format.

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
            messages=[{"role": msg.role, "content": msg.content[0].get("text")} for msg in messages],  # type: ignore
            response_format={"type": "json_object"} if is_json_response else None,
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
