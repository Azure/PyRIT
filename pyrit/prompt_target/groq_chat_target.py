# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
from pyrit.prompt_target.openai.openai_chat_target import OpenAIChatTarget
from pyrit.models import ChatMessageListDictContent
from pyrit.exceptions import PyritException, EmptyResponseException
from openai.types.chat import ChatCompletion
from pyrit.exceptions import pyrit_target_retry

logger = logging.getLogger(__name__)

class GroqChatTarget(OpenAIChatTarget):

    @pyrit_target_retry
    async def _complete_chat_async(self, messages: list[ChatMessageListDictContent], is_json_response: bool) -> str:
        """
        Completes asynchronous chat request.

        Sends a chat message to the OpenAI chat model and retrieves the generated response.

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
