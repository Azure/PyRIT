# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Optional, Sequence, Dict, Union

from pyrit.prompt_target import PromptChatTarget
from pyrit.score.scorer import Scorer
from pyrit.models import PromptRequestPiece, SeedPrompt
from pyrit.models.score import Score, UnvalidatedScore

# political_general_scorer = SelfAskGeneralScorer(
#     chat_target=self_ask_target,
#     system_prompt="You're a helpful assistant. Give me a value, here are examples, the format looks like this",
#     prompt_fstring_format='prompt to evaluate {} task is {}',
#     scorer_type = "float_scale", # float_scale or true_false
#     score_value_json_key = "output",
#     category=["misin"]
#     labels=[],
# )
 
class SelfAskGeneralScorer(Scorer):
    def __init__(self, chat_target: PromptChatTarget, 
                 system_prompt:str = None, prompt_fstring_format: str = None, 
                 scorer_type: str = None, score_value_json_key: str = None, 
                 category:list =None, labels: list=None):
        self._prompt_target = chat_target
        self._system_prompt = system_prompt
        self.prompt_fstring_format = prompt_fstring_format
        self.scorer_type = scorer_type
        self.score_value_json_key = score_value_json_key
        self._score_category = category
        self.labels = labels

        print("SelfAskGeneralScorer initialized")

    async def score_async(self, request_response: PromptRequestPiece, *, task: Optional[str] = None) -> list[Score]:
        self.validate(request_response, task=task)

        if self.scorer_type == "true_false":
            print("True False scorer type")

            unvalidated_score: UnvalidatedScore = await self._score_value_with_llm(
                prompt_target=self._prompt_target,
                system_prompt=self._system_prompt,
                prompt_request_value=request_response.converted_value,
                prompt_request_data_type=request_response.converted_value_data_type,
                scored_prompt_id=request_response.id,
                category=self._score_category,
                task=task,
                orchestrator_identifier=request_response.orchestrator_identifier,
            )
            score = unvalidated_score.to_score(score_value=unvalidated_score.raw_score_value)

            # self._memory.add_scores_to_memory(scores=[score])
            return [score]
        return []

    
    def validate(self, request_response: PromptRequestPiece, *, task: Optional[str] = None) -> None:
        pass
