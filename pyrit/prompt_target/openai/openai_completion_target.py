# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
from openai import NotGiven, NOT_GIVEN
from openai.types.completion import Completion
from typing import Optional

from pyrit.models import PromptResponse, PromptRequestResponse, construct_response_from_request
from pyrit.prompt_target import limit_requests_per_minute, OpenAITarget


logger = logging.getLogger(__name__)


class OpenAICompletionTarget(OpenAITarget):

    def __init__(
        self,
        max_tokens: Optional[int] | NotGiven = NOT_GIVEN,
        temperature: float = 1.0,
        top_p: float = 1.0,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        *args,
        **kwargs,
    ):
        """
        Args:
            max_tokens (int, Optional): The maximum number of tokens that can be generated in the
              completion. The token count of your prompt plus `max_tokens` cannot exceed the model's
              context length.
        """

        super().__init__(*args, **kwargs)

        self._max_tokens = max_tokens
        self._temperature = temperature
        self._top_p = top_p
        self._frequency_penalty = frequency_penalty
        self._presence_penalty = presence_penalty

    def _set_azure_openai_env_configuration_vars(self):
        self.deployment_environment_variable = "AZURE_OPENAI_COMPLETION_DEPLOYMENT"
        self.endpoint_uri_environment_variable = "AZURE_OPENAI_COMPLETION_ENDPOINT"
        self.api_key_environment_variable = "AZURE_OPENAI_COMPLETION_KEY"

    @limit_requests_per_minute
    async def send_prompt_async(self, *, prompt_request: PromptRequestResponse) -> PromptRequestResponse:
        """
        Sends a normalized prompt async to the prompt target.
        """
        self._validate_request(prompt_request=prompt_request)
        request = prompt_request.request_pieces[0]

        logger.info(f"Sending the following prompt to the prompt target: {request}")

        text_response: Completion = await self._async_client.completions.create(
            model=self._deployment_name,
            prompt=request.converted_value,
            top_p=self._top_p,
            temperature=self._temperature,
            frequency_penalty=self._frequency_penalty,
            presence_penalty=self._presence_penalty,
            max_tokens=self._max_tokens,
        )
        prompt_response = PromptResponse(
            completion=text_response.choices[0].text,
            prompt=request.converted_value,
            id=text_response.id,
            completion_tokens=text_response.usage.completion_tokens,
            prompt_tokens=text_response.usage.prompt_tokens,
            total_tokens=text_response.usage.total_tokens,
            model=text_response.model,
            object=text_response.object,
        )
        response_entry = construct_response_from_request(
            request=request,
            response_text_pieces=[prompt_response.completion],
            prompt_metadata=prompt_response.to_json(),
        )

        return response_entry

    def _validate_request(self, *, prompt_request: PromptRequestResponse) -> None:
        if len(prompt_request.request_pieces) != 1:
            raise ValueError("This target only supports a single prompt request piece.")

        if prompt_request.request_pieces[0].converted_value_data_type != "text":
            raise ValueError("This target only supports text prompt input.")

        request = prompt_request.request_pieces[0]
        messages = self._memory.get_chat_messages_with_conversation_id(conversation_id=request.conversation_id)

        if len(messages) > 0:
            raise ValueError("This target only supports a single turn conversation.")
