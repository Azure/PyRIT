# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import asyncio
import logging

from typing import Optional

from pyrit.memory import MemoryInterface
from pyrit.models.prompt_request_piece import PromptDataType, PromptRequestPiece
from pyrit.models.prompt_request_response import PromptRequestResponse
from pyrit.orchestrator import Orchestrator
from pyrit.prompt_normalizer import PromptNormalizer
from pyrit.prompt_normalizer.normalizer_request import NormalizerRequest
from pyrit.prompt_target import PromptTarget
from pyrit.prompt_converter import PromptConverter
from pyrit.score import Scorer

logger = logging.getLogger(__name__)


class PromptSendingOrchestrator(Orchestrator):
    """
    This orchestrator takes a set of prompts, converts them using the list of PromptConverters,
    scores them with the provided scorer, and sends them to a target.
    """

    def __init__(
        self,
        prompt_target: PromptTarget,
        prompt_converters: Optional[list[PromptConverter]] = None,
        scorer: Optional[Scorer] = None,
        memory: MemoryInterface = None,
        batch_size: int = 10,
        verbose: bool = False,
    ) -> None:
        """
        Args:
            prompt_target (PromptTarget): The target for sending prompts.
            prompt_converters (list[PromptConverter], optional): List of prompt converters. These are stacked in
                the order they are provided. E.g. the output of converter1 is the input of converter2.
            prompt_response_scorer (Scorer, optional): Scorer to use for each prompt request response, to be
                scored immediately after recieving response. Default is None.
            memory (MemoryInterface, optional): The memory interface. Defaults to None.
            batch_size (int, optional): The (max) batch size for sending prompts. Defaults to 10.
        """
        super().__init__(
            prompt_converters=prompt_converters,
            scorer=scorer,
            memory=memory,
            verbose=verbose)

        self._prompt_normalizer = PromptNormalizer(memory=self._memory)

        self._prompt_target = prompt_target
        self._prompt_target._memory = self._memory

        self._batch_size = batch_size

    async def send_prompts_async(
        self, *, prompt_list: list[str], prompt_type: PromptDataType = "text"
    ) -> list[PromptRequestResponse]:
        """
        Sends the prompts to the prompt target.
        """

        requests: list[NormalizerRequest] = []
        for prompt in prompt_list:
            requests.append(
                self._create_normalizer_request(
                    prompt_text=prompt,
                    prompt_type=prompt_type,
                    converters=self._prompt_converters,
                )
            )

        return await self.send_normalizer_requests_async(
            prompt_request_list=requests,
        )

    async def send_normalizer_requests_async(
        self, *, prompt_request_list: list[NormalizerRequest]
    ) -> list[PromptRequestResponse]:
        """
        Sends the normalized prompts to the prompt target.
        """
        for request in prompt_request_list:
            request.validate()

        responses = await self._prompt_normalizer.send_prompt_batch_to_target_async(
            requests=prompt_request_list,
            target=self._prompt_target,
            labels=self._global_memory_labels,
            orchestrator_identifier=self.get_identifier(),
            batch_size=self._batch_size,
        )
    
        response_pieces_to_score = []
        if self._scorer:
            for response in responses:
                for piece in response.request_pieces:
                    if piece.role == "assistant":
                        # Add to a list of responses to score
                        response_pieces_to_score.append(piece)
                        # self._scorer.score_async(piece.converted_value) # Note this is adding score to the memory in another (score) table

                        # TODO: Add batch support to score here, would this go into the scorer class?
                        # TODO: Maybe consider having this on the PromptRequestPiece object instead --> this could be in another story to correspond
                            # If we implement this then we won't need the score table in DB
                        # Orchestrator.get_memory(score_table) -- this is how we interact with the results

        self._score_prompts_batch_async(prompts=response_pieces_to_score, scorer=self._scorer) # This will add scores to the memory in the ScoreEntries table
        # These should be correlated by PromptRequestResponseID
        # TODO: Figure out how to extract these in a way that's demoable and where you can cross check the scoring with the PromptRequestResponse object --
        # Maybe this should be a helper function to get these into the same table or a follow-up story?

        return responses
    
    # Note: These are functions within scoring_orchestrator...think about if they belong here?
    # This is modified to not return the score objects, and instead to just chunk and score
    async def _score_prompts_batch_async(self, prompts: list[PromptRequestPiece], scorer: Scorer) -> None:
        results = []

        for prompts_batch in self._chunked_prompts(prompts, self._batch_size):
            tasks = []
            for prompt in prompts_batch:
                tasks.append(scorer.score_async(request_response=prompt))

            batch_results = await asyncio.gather(*tasks)
            results.extend(batch_results)

        # results is a list[list[str]] and needs to be flattened
        # return [score for sublist in results for score in sublist]
    
    def _chunked_prompts(self, prompts, size):
        for i in range(0, len(prompts), size):
            yield prompts[i : i + size]
            

