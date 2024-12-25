# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import enum
import json
import logging
from typing import Optional

from pyrit.common import net_utility
from pyrit.models import PromptRequestResponse, construct_response_from_request
from pyrit.prompt_target import PromptTarget, limit_requests_per_minute

logger = logging.getLogger(__name__)


class GandalfLevel(enum.Enum):
    LEVEL_1 = "baseline"
    LEVEL_2 = "do-not-tell"
    LEVEL_3 = "do-not-tell-and-block"
    LEVEL_4 = "gpt-is-password-encoded"
    LEVEL_5 = "word-blacklist"
    LEVEL_6 = "gpt-blacklist"
    LEVEL_7 = "gandalf"
    LEVEL_8 = "gandalf-the-white"
    LEVEL_9 = "adventure-1"
    LEVEL_10 = "adventure-2"


class GandalfTarget(PromptTarget):

    def __init__(
        self,
        *,
        level: GandalfLevel,
        max_requests_per_minute: Optional[int] = None,
    ) -> None:
        super().__init__(max_requests_per_minute=max_requests_per_minute)

        self._endpoint = "https://gandalf.lakera.ai/api/send-message"
        self._defender = level.value

    @limit_requests_per_minute
    async def send_prompt_async(self, *, prompt_request: PromptRequestResponse) -> PromptRequestResponse:
        self._validate_request(prompt_request=prompt_request)
        request = prompt_request.request_pieces[0]

        logger.info(f"Sending the following prompt to the prompt target: {request}")

        response = await self._complete_text_async(request.converted_value)

        response_entry = construct_response_from_request(request=request, response_text_pieces=[response])

        return response_entry

    def _validate_request(self, *, prompt_request: PromptRequestResponse) -> None:
        if len(prompt_request.request_pieces) != 1:
            raise ValueError("This target only supports a single prompt request piece.")

        if prompt_request.request_pieces[0].converted_value_data_type != "text":
            raise ValueError("This target only supports text prompt input.")

    async def check_password(self, password: str) -> bool:
        """
        Checks if the password is correct

        True means the password is correct, False means it is not
        """
        payload: dict[str, object] = {
            "defender": self._defender,
            "password": password,
        }

        resp = await net_utility.make_request_and_raise_if_error_async(
            endpoint_uri=self._endpoint, method="POST", request_body=payload, post_type="data"
        )

        if not resp.text:
            raise ValueError("The chat returned an empty response.")

        json_response = resp.json()
        return json_response["success"]

    async def _complete_text_async(self, text: str) -> str:
        payload: dict[str, object] = {
            "defender": self._defender,
            "prompt": text,
        }

        resp = await net_utility.make_request_and_raise_if_error_async(
            endpoint_uri=self._endpoint, method="POST", request_body=payload, post_type="data"
        )

        if not resp.text:
            raise ValueError("The chat returned an empty response.")

        answer = json.loads(resp.text)["answer"]

        logger.info(f'Received the following response from the prompt target "{answer}"')
        return answer
