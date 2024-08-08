# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging

from typing import Optional

from pyrit.memory import MemoryInterface
from pyrit.models.prompt_request_piece import PromptDataType
from pyrit.models.prompt_request_response import PromptRequestResponse
from pyrit.orchestrator import Orchestrator
from pyrit.orchestrator.scoring_orchestrator import ScoringOrchestrator
from pyrit.prompt_normalizer import PromptNormalizer
from pyrit.prompt_normalizer.normalizer_request import NormalizerRequest
from pyrit.prompt_target import PromptTarget
from pyrit.prompt_converter import PromptConverter
from pyrit.score import Scorer


logger = logging.getLogger(__name__)


class PromptSendingOrchestrator(Orchestrator):
    """
    This orchestrator takes a set of prompts, converts them using the list of PromptConverters,
    sends them to a target, and scores the resonses with scorers (if provided).
    """

    def __init__(
        self,
        prompt_target: PromptTarget,
        prompt_converters: Optional[list[PromptConverter]] = None,
        scorers: Optional[list[Scorer]] = None,
        memory: MemoryInterface = None,
        batch_size: int = 10,
        verbose: bool = False,
    ) -> None:
        """
        Args:
            prompt_target (PromptTarget): The target for sending prompts.
            prompt_converters (list[PromptConverter], optional): List of prompt converters. These are stacked in
                the order they are provided. E.g. the output of converter1 is the input of converter2.
            scorers (list[Scorer], optional): List of scorers to use for each prompt request response, to be
                scored immediately after recieving response. Default is None.
            memory (MemoryInterface, optional): The memory interface. Defaults to None.
            batch_size (int, optional): The (max) batch size for sending prompts. Defaults to 10.
        """
        super().__init__(prompt_converters=prompt_converters, memory=memory, verbose=verbose)

        self._scorers = scorers

        self._prompt_target = prompt_target
        self._prompt_target._memory = self._memory

        self._batch_size = batch_size

    async def send_prompts_async(
        self, 
        *, 
        prompt_list: list[str], 
        prompt_type: PromptDataType = "text", 
        metadata: str | None = None
    ) -> list[PromptRequestResponse]:
        """
        Sends the prompts to the prompt target.
        """

        # This happens often when trying to send one prompt, but forgetting to wrap it in brackets.
        if not isinstance(prompt_list, list):
            logger.warning(f"Warning: passed {type(prompt_list)} to PromptSendingOrchestrator." \
                           "Did you mean to pass a list?")

        requests: list[NormalizerRequest] = []
        for prompt in prompt_list:
            requests.append(
                self._create_normalizer_request(
                    prompt_text=prompt,
                    prompt_type=prompt_type,
                    converters=self._prompt_converters,
                    metadata=metadata if metadata else None # NOTE added
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

        # Normalizer is responsible for storing the requests in memory
        # The labels parameter may allow me to stash class information for each kind of prompt.
        responses: list[PromptRequestResponse] = await self._prompt_normalizer.send_prompt_batch_to_target_async(
            requests=prompt_request_list,
            target=self._prompt_target,
            labels=self._global_memory_labels,
            orchestrator_identifier=self.get_identifier(),
            batch_size=self._batch_size,
        )

        # These are the responses from the target
        # print(responses[0].request_pieces)

        if self._scorers:
            response_ids = []
            for response in responses:
                for piece in response.request_pieces:
                    id = str(piece.id)
                    response_ids.append(id)

            await self._score_responses_async(response_ids)

        return responses

    async def _score_responses_async(self, prompt_ids: list[str]):
        with ScoringOrchestrator(
            memory=self._memory,
            batch_size=self._batch_size,
            verbose=self._verbose,
        ) as scoring_orchestrator:
            for scorer in self._scorers:
                await scoring_orchestrator.score_prompts_by_request_id_async(
                    scorer=scorer,
                    prompt_ids=prompt_ids,
                    responses_only=True,
                )
