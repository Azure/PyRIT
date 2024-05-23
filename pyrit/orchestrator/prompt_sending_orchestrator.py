# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import asyncio
import logging

from typing import Optional

from pyrit.memory import MemoryInterface
from pyrit.models.prompt_request_piece import PromptDataType, PromptRequestPiece
from pyrit.models.prompt_request_response import PromptRequestResponse
from pyrit.orchestrator import Orchestrator
from pyrit.orchestrator.scoring_orchestrator import ScoringOrchestrator
from pyrit.prompt_normalizer import PromptNormalizer
from pyrit.prompt_normalizer.normalizer_request import NormalizerRequest
from pyrit.prompt_target import PromptTarget
from pyrit.prompt_converter import PromptConverter
from pyrit.score import Scorer, Score


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
        scorer_list: Optional[list[Scorer]] = None,
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
        super().__init__(prompt_converters=prompt_converters, memory=memory, verbose=verbose)

        self._prompt_normalizer = PromptNormalizer(memory=self._memory)
        self._scorer_list = scorer_list

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

        if self._scorer_list:
            with ScoringOrchestrator() as scoring_orchestrator:
                for scorer in self._scorer_list:
                    await scoring_orchestrator.score_prompts_by_orchestrator_id_async(
                        scorer=scorer,
                        orchestrator_ids=[self.get_identifier()["id"]],
                    )

            # TODO: Maybe consider having this on the PromptRequestPiece object instead --> this could be in another story to correspond
            # If we implement this then we won't need the score table in DB

        return responses

    def get_score_memory(self):
        """
        Retrieves the scores of the PromptRequestPieces associated with this orchestrator.
        These exist if a scorer is provided to the orchestrator.
        """
        return self._memory.get_scores_by_orchestrator_id(orchestrator_id=id(self))
