import logging
import re

from typing import Literal, Union, List
from pyrit.prompt_target import PromptTarget
from pyrit.memory import MemoryInterface
from pyrit.common import default_values
from pyrit.common import net_utility as nu
from pyrit.models import (
    construct_response_from_request,
    PromptRequestResponse, 
)

logger = logging.getLogger(__name__)

PromptShieldEntryKind = Literal[None, "userPrompt", "documents"]

class PromptShieldTarget(PromptTarget):

    ### ATTRIBUTES ###
    ENDPOINT_URI_ENVIRONMENT_VARIABLE: str = "AZURE_CONTENT_SAFETY_ENDPOINT"
    API_KEY_ENVIRONMENT_VARIABLE: str = "AZURE_CONTENT_SAFETY_KEY"
    _endpoint: str
    _api_key: str
    _force_entry_kind: PromptShieldEntryKind

    ### METHODS ###
    def __init__(
            self,
            endpoint: str,
            api_key: str,
            field: PromptShieldEntryKind = None,
            memory: Union[MemoryInterface, None] = None
        ) -> None:

        super().__init__(memory=memory)

        self._endpoint = default_values.get_required_value(
            env_var_name=self.ENDPOINT_URI_ENVIRONMENT_VARIABLE, passed_value=endpoint
        )

        self._api_key = default_values.get_required_value(
            env_var_name=self.API_KEY_ENVIRONMENT_VARIABLE, passed_value=api_key
        )

        self._force_entry_kind: PromptShieldEntryKind = field
        '''
        this is for the "force_entry_kind" parameter.
        the default behavior is for PromptShieldTarget to parse the input it's given using regex.
        for example, if the input string is:
        'hello world! <document> document1 </document> <document> document2 </document>
        then the target will send this to the Prompt Shield endpoint:
        userPrompt: 'hello world!'
        documents: ['document1', 'document2']

        None is the default state (use parsing). userPrompt and document are the other states, and
        you can use those to force only one parameter (either userPrompt or documents) to be populated
        with the raw input (no parsing, regex, etc.)
        '''

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

        userPrompt, docsList = self._input_parser(request.original_value)

        body = {
            'userPrompt': userPrompt,
            'documents': docsList
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
                    "Sorry, but requests with multiple entries are not supported. " \
                    "Please wrap each PromptRequestPiece in a PromptRequestResponse." \
                )
        if type(prompt_request.request_pieces[0].original_value) != str:
            raise ValueError("The content of this request must be a string.")
        
    def _input_parser(self, input_str: str) -> tuple[str, list[str]]:
        '''
        Parses the input given to the target to extract the two fields sent to
        Prompt Shield:
        userPrompt: str
        and
        documents: list[str]
        '''

        match self._force_entry_kind:
            case 'userPrompt':
                return (input_str, [""])
            case 'documents':
                return ("", [input_str])
            case _:
                userPrompt: str = ""
                docsList: list[str] = [""]

                userPrompt_pattern: str = r"^(.*?)<"
                docsList_pattern: str = r"<document>(.*?)</document>"

                userPrompt_match = re.search(userPrompt_pattern, input_str)
                if userPrompt_match:
                    userPrompt = userPrompt_match.group(1).strip()

                docsList = re.findall(docsList_pattern, input_str)

                return (userPrompt, docsList)