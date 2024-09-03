import logging
import json
import re 
import enum

from typing import Optional

from pyrit.memory import DuckDBMemory, MemoryInterface
from pyrit.models import Score
from pyrit.models import PromptRequestPiece, PromptRequestResponse, construct_response_from_request
from pyrit.prompt_target import PromptChatTarget, PromptTarget
from pyrit.score.scorer import Scorer
from pyrit.common import net_utility

logger = logging.getLogger(__name__)


class GandalfTongueTiedLevel(enum.Enum):
    LEVEL_1 = "adventure-8"
    LEVEL_2 = "adventure-9"
    LEVEL_3 = "adventure-10"
    LEVEL_4 = "adventure-11"
    LEVEL_5 = "adventure-12"



class GandalfTongueTiedTarget(PromptTarget):

    def __init__(
        self,
        *,
        level: GandalfTongueTiedLevel,
        memory: MemoryInterface = None,
    ) -> None:
        self._memory = memory if memory else DuckDBMemory()

        self._endpoint = "https://gandalf.lakera.ai/api/send-message"
        self._defender = level.value


    async def send_prompt_async(self, *, prompt_request: PromptRequestResponse) -> PromptRequestResponse:
        self._validate_request(prompt_request=prompt_request)
        request = prompt_request.request_pieces[0]

        logger.info(f"Sending the following prompt to the prompt target: {request}")

        responses = await self._complete_text_async(request.converted_value)
        response_entry = construct_response_from_request(request=request, response_text_pieces=responses)

        return response_entry


    def _validate_request(self, *, prompt_request: PromptRequestResponse) -> None:
        if len(prompt_request.request_pieces) != 1:
            raise ValueError("This target only supports a single prompt request piece.")

        if prompt_request.request_pieces[0].converted_value_data_type != "text":
            raise ValueError("This target only supports text prompt input.")


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

        # TODO focus on one LM for now
        responses = parse_responses(answer)

        return responses


