import logging
import uuid
import json
from typing import Literal, Union, List

from pyrit.prompt_target import PromptShieldTarget
from pyrit.memory import MemoryInterface, DuckDBMemory
from pyrit.models import PromptRequestResponse, PromptRequestPiece, Score
from pyrit.score import Scorer

logger = logging.getLogger(__name__)

class PromptShieldScorer(Scorer):
    """
    Returns true if an attack or jailbreak has been detected by Prompt Shield. 
    """

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
        
        self.scorer_type = "true_false"
        self._conversation_id = str(uuid.uuid4())
        self._memory = memory if memory else DuckDBMemory()
        self._target: PromptShieldTarget = target

    async def score_async(self, request_response: PromptRequestPiece) -> List[Score]:
        self.validate(request_response=request_response)

        body = request_response.request_pieces[0].original_value

        request = PromptRequestResponse(
            [
                PromptRequestPiece(
                    role="user",
                    original_value=body,
                    prompt_metadata= request_response.request_pieces[0].prompt_metadata,
                    conversation_id=self._conversation_id,
                    prompt_target_identifier=self._target.get_identifier()
                )
            ]
        )

        # The body of the Prompt Shield response
        response = await self._target.send_prompt_async(prompt_request=request)


        response: str = response.request_pieces[0].original_value

        # Whether or not any of the documents or userPrompt got flagged as an attack
        result: bool = any(self._parse_response_to_boolean_list(response))
        
        score = Score(
            score_type='true_false',
            score_value=result,
            score_value_description=None,
            score_category='attack_detection',
            score_metadata=response,
            score_rationale=None,
            scorer_class_identifier=self.get_identifier(),
            prompt_request_response_id=request_response.request_pieces[0].id # TODO: this is causing errors
        )

        self._memory.add_scores_to_memory(scores=[score])
        return [score]

    def _parse_response_to_boolean_list(self, response: str) -> list[bool]:
        """
        Remember that you can just access the metadata attribute to get the original Prompt Shield endpoint response,
        and then just call json.loads() on it to interact with it.
        """

        response_json:dict = json.loads(response)

        # A list containing one entry, which is a boolean for whether Prompt Shield detected an attack or not
        # in the user prompt.
        user_prompt_attack_detected: bool = [response_json['userPromptAnalysis']['attackDetected']]

        # A list containing n-many entries, where each entry is a boolean, for the n-many documents sent to
        # Prompt Shield, and each entry corresponds to if an attack was detected.
        documents_list_attack_detected: list[bool] = [entry['attackDetected'] for entry in response_json['documentsAnalysis']]
        
        return user_prompt_attack_detected + documents_list_attack_detected
    
    def validate(
            self, 
            request_response: PromptRequestPiece
        ) -> None:
        """
        TODO: Writeup
        """
        pass

