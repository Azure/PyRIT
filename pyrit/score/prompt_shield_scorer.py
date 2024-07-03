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
    A scorer that uses Prompt Shield as a target.
    Combine this with a scoring template.
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

        self.validate(request_response=request_response)

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
