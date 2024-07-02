import requests
import os
import logging
import uuid
import duckdb

from typing import Any, Coroutine, Literal, Union, List
from pyrit.prompt_target import PromptTarget
from pyrit.memory import MemoryInterface, DuckDBMemory

from pyrit.common import default_values
from pyrit.common import net_utility as nu

from pyrit.models import (
    data_serializer_factory, 
    construct_response_from_request,
    PromptRequestResponse, 
    PromptRequestPiece,
    Score
)

from pyrit.score import (
    Scorer
)

logger = logging.getLogger(__name__)

PromptShieldEntryKind = Literal["userPrompt", "document"]

class PromptShieldTarget(PromptTarget):
    '''
    Prompt Shield as a Target. This uses the same logic as the Prompt Shield Scorer.
    TODO: Error catching. This includes checking for the most up-to-date API version for Prompt Shield,
    which is passed as parameters (params) in the HTTP request.
    TODO: Unit tests.
    TODO: The current implementation makes you specify which field you want to use when creating the
    target, since Prompt Shield has both a userPrompt and a document field.
    This doesn't account for cases in which you want to split the input across the two fields.
    I'm not sure how to implement this in orchestration, because every orchestrator has a slightly
    different flow that might affect the field(s) usage differently.
    '''

    ### ATTRIBUTES ###
    ENDPOINT_URI_ENVIRONMENT_VARIABLE: str = "AZURE_CONTENT_SAFETY_ENDPOINT"
    API_KEY_ENVIRONMENT_VARIABLE: str = "AZURE_CONTENT_SAFETY_KEY"
    _endpoint: str
    _api_key: str
    _field: PromptShieldEntryKind

    ### METHODS ###
    def __init__(
            self,
            endpoint: str,
            api_key: str,
            field: PromptShieldEntryKind = 'document',
            memory: Union[MemoryInterface, None] = None
        ) -> None:

        super().__init__(memory=memory)

        self._endpoint = default_values.get_required_value(
            env_var_name=self.ENDPOINT_URI_ENVIRONMENT_VARIABLE, passed_value=endpoint
        )

        self._api_key = default_values.get_required_value(
            env_var_name=self.API_KEY_ENVIRONMENT_VARIABLE, passed_value=api_key
        )

        self._field = field

    async def send_prompt_async(self, *, prompt_request: PromptRequestResponse) -> PromptRequestResponse:
        self._validate_request(prompt_request=prompt_request)

        request = prompt_request.request_pieces[0]

        logger.info(f"Sending the following prompt to the prompt target: {request}")

        headers = {
            'Ocp-Apim-Subscription-Key': self._api_key,
            'Content-Type': 'application/json',
        }

        params = {
            'api-version': '2024-02-15-preview',
        }

        # You need to send something for every field, even if said field is empty.
        userPromptValue: str = ""
        documentsValue: List[str] = [""]

        # match request.prompt_metadata:
        match self._field:
            case 'userPrompt':
                userPromptValue = request.converted_value
            case 'document':
                documentsValue = [request.converted_value]

        body = {
            'userPrompt': userPromptValue,
            'documents': documentsValue
        }

        response = await nu.make_request_and_raise_if_error_async(
            endpoint_uri=f'{self._endpoint}/contentsafety/text:shieldPrompt',
            method='POST',
            params=params,
            headers=headers,
            request_body=body
        )

        logger.info("Received a valid response from the prompt target")

        data = response.content

        response_entry = construct_response_from_request(
            request=request, response_text_pieces=[str(data)], response_type="text"
        )

        return response_entry

    
    def _validate_request(self, *, prompt_request: PromptRequestResponse) -> None:
        if len(prompt_request.request_pieces) == 0:
            raise ValueError("This target requires at least one prompt request piece.")
        if len(prompt_request.request_pieces) > 1:
            raise ValueError(
                    "Sorry, but requests with multiple entries are not supported yet. " \
                    "Please wrap each PromptRequestPiece in a PromptRequestResponse." \
                )