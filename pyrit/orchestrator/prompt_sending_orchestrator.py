# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging

from typing import Optional

from pyrit.memory import MemoryInterface
from pyrit.models.prompt_request_piece import PromptDataType
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
    
        if self._scorer:
            for response in responses:
                for piece in response.request_pieces:
                    if piece.role == "assistant":
                        response_data_type = piece.converted_value_data_type
                
                        # TODO: This is a list...are we assuming there could be multiple assistant responses?
                        # e.g. for multiturn? Is this orchestrator single-turn only?

                        if response_data_type == "text":
                            score_func = self._scorer.score_text_async
                        elif response_data_type == "image_path":
                            score_func = self._scorer.score_image_async
                        else:
                            raise ValueError(f"Cannot score unsupported response data type of: {response_data_type}")
                        
                        response.score = score_func(piece.converted_value) # TODO: Assumes only one assitant request piece per PromptRequestResponse...
                        # TODO: Maybe consider having this on the PromptRequestPiece object instead

                        print(response.score)

        return responses
            

