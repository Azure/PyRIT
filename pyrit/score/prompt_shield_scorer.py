import logging
import uuid

from typing import Literal, Union, List
from pyrit.prompt_target import PromptShieldTarget
from pyrit.memory import MemoryInterface, DuckDBMemory


from pyrit.models import (
    PromptRequestResponse, 
    PromptRequestPiece,
    Score
)

from pyrit.score import (
    Scorer
)

logger = logging.getLogger(__name__)

PromptShieldEntryKind = Literal["userPrompt", "document"]

class PromptShieldScorer(Scorer):
    '''
    TODO list
    TODO: Implementation (Foreign Key Errors)
    TODO: Local testing
    TODO: Unit testing

    (Thanks to Richard Lundeen for this idea!)
    A scorer which returns a boolean value for detection by Prompt Shield.
    Since there's one scorer entry per prompt entry:
    (1 Scorer Entry == 1 PromptPiece == 1 HTTP Request to Prompt Shield)
    The score applies to whichever field (userPrompt or document/s) was sent in the
    PromptRequestPiece.

    NOTE: The HTTP body as a JSON returns a boolean value, and the intent is for it
    to be stored as a boolean value in the scoring table. If it can't, it will be
    converted to a string literal.
    '''

    ### ATTRIBUTES ###
    scorer_type: str
    _conversation_id: str
    _memory: Union[MemoryInterface, None]
    _target: PromptShieldTarget

    ### METHODS ###
    def __init__(
            self,
            target: PromptShieldTarget,
            memory: Union[MemoryInterface, None] = None
        ) -> None:
        '''
        TODO: description
        '''
        
        self.scorer_type = "true_false"
        self._conversation_id = str(uuid.uuid4())
        self._memory = memory if memory else DuckDBMemory()
        self._target: PromptShieldTarget = target

    async def score_async(self, request_response: PromptRequestPiece) -> List[Score]:
        '''
        NOTE: Use this as a debugging entry point
        TODO: description
        '''
        self.validate(request_response=request_response)

        # TODO: Fix this. The converted value should be the boolean for attack detection,
        # while the original would be the JSON response from the HTTP body.

        # body = request_response.converted_value
        body = request_response.original_value

        request = PromptRequestResponse(
            [
                PromptRequestPiece(
                    role="user",
                    original_value=body,
                    prompt_metadata=request_response.prompt_metadata,
                    conversation_id=self._conversation_id,
                    prompt_target_identifier=self._target.get_identifier()
                )
            ]
        )

        response = await self._target.send_prompt_async(prompt_request=request)

        # TODO: Fix this. The converted value should be the boolean for attack detection,
        # while the original would be the JSON response from the HTTP body.

        # result = response.request_pieces[0].converted_value

        # NOTE: This is a hacky workaround for the MVP. This needs to be fixed
        # in the PromptShieldTarget.

        result = str(response.request_pieces[0].original_value).split('"attackDetected":')[-1].split("}]}")[0]
        
        score = Score(
            score_type='true_false',
            score_value=result,
            score_value_description=None,
            score_category='attack_detection',
            score_metadata=None,
            score_rationale=None,
            scorer_class_identifier=self.get_identifier(),
            prompt_request_response_id=request_response.id # TODO: this is causing errors
        )

        self._memory.add_scores_to_memory(scores=[score])
        return [score]
    
    def validate(
            self, 
            request_response: PromptRequestPiece
        ) -> None:
        '''
        NOTE: Use this as a debugging entry point
        TODO: implementation
        TODO: description
        '''
        pass
