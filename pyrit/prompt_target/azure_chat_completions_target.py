# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
from typing import Optional
from azure.ai.inference.aio import ChatCompletionsClient
from azure.core.credentials import AzureKeyCredential

from azure.identity.aio import DefaultAzureCredential

from pyrit.common import default_values
from pyrit.models import (
    PromptRequestPiece,
    PromptRequestResponse,
    construct_response_from_request,
)

from pyrit.models.score import Score
from pyrit.prompt_target import PromptChatTarget, limit_requests_per_minute

logger = logging.getLogger(__name__)


class AzureChatCompletionsTarget(PromptChatTarget):

    ENDPOINT_URI_ENVIRONMENT_VARIABLE = "AZURE_CHAT_COMPLETIONS_ENDPOINT"
    API_KEY_ENVIRONMENT_VARIABLE = "AZURE_CHAT_COMPLETIONS_KEY"

    def __init__(
        self,
        *,
        endpoint: str = None,
        api_key: str = None,
        use_aad_auth: bool = False,
        max_requests_per_minute: Optional[int] = None,
        frequency_penalty: Optional[float] = None,
        presence_penalty: Optional[float] = None,
        max_tokens: Optional[int] = 8192,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        seed: Optional[int] = None,
    ) -> None:

        PromptChatTarget.__init__(self, max_requests_per_minute=max_requests_per_minute)

        endpoint = endpoint or default_values.get_required_value(
            env_var_name=self.ENDPOINT_URI_ENVIRONMENT_VARIABLE, passed_value=endpoint
        )

        api_key = api_key or default_values.get_required_value(
            env_var_name=self.API_KEY_ENVIRONMENT_VARIABLE, passed_value=api_key
        )

        if use_aad_auth:
            credential=DefaultAzureCredential()
        else:
            credential=AzureKeyCredential(api_key)

        self._client = ChatCompletionsClient(
            endpoint=endpoint,
            credential=credential,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            seed=seed,
        )


    @limit_requests_per_minute
    async def send_prompt_async(self, *, prompt_request: PromptRequestResponse) -> PromptRequestResponse:

        self._validate_request(prompt_request=prompt_request)
        request_piece: PromptRequestPiece = prompt_request.request_pieces[0]

        is_json_response = self.is_response_format_json(request_piece)

        messages = self._memory.get_chat_messages_with_conversation_id(conversation_id=request_piece.conversation_id)
        messages.append(request_piece.to_chat_message())

        logger.info(f"Sending the following prompt to the prompt target: {self} {request_piece}")

        payload = {
            "messages": [msg.dict(exclude_none=True) for msg in messages],
        }

        response = await self._client.complete(
            body=payload,
            response_format="json_object" if is_json_response else "text",
        )

        response_text = response.choices[0].message.content
        azure_filter_scores = response.choices[0].get("content_filter_results")
        if azure_filter_scores:
            self._add_scores(azure_filter_scores=azure_filter_scores, prompt_request=request_piece)
        finish_reason = response.choices[0]["finish_reason"] # should be stop if things were a success

        response_type = "text"

        if finish_reason != "stop":
            response_type = "error"

        logger.info(f'Received the following response from the prompt target "{response_text}"')

        return construct_response_from_request(request=request_piece, response_text_pieces=[response_text], response_type=response_type)


    # The response contains scores that are automatically detected by Azure. We add these scores to the memory.
    def _add_scores(self, *, azure_filter_scores, prompt_request):
        scores = []
        for key, value in azure_filter_scores.items():
            score_value = value.get("detected", value.get("filtered", False))
            if score_value != None:
                score = Score(
                    score_value=str(score_value),
                    score_value_description="Detected automatically from Azure Completion Response",
                    score_type="true_false",
                    score_category=key,
                    score_rationale="",
                    score_metadata="",
                    scorer_class_identifier=str(self.get_identifier()),
                    prompt_request_response_id=prompt_request.id,
                )

                scores.append(score)

        if scores:
            # need to insert the prompt first since it has a primary key constraint
            # this will be re-inserted by prompt_normalizer, but that's okay
            self._memory.add_request_pieces_to_memory(request_pieces=[prompt_request])
            self._memory.add_scores_to_memory(scores=scores)



    def _validate_request(self, *, prompt_request: PromptRequestResponse) -> None:
        pass

    def is_json_response_supported(self) -> bool:
        """Indicates that this target supports JSON response format."""
        return True
