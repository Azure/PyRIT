# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import abc
import asyncio
import concurrent.futures
from uuid import uuid4
from pyrit.memory import MemoryInterface
from pyrit.models import PromptRequestResponse, PromptRequestPiece
from pyrit.prompt_normalizer.normalizer_request import NormalizerRequest
from pyrit.prompt_target import PromptTarget


class PromptNormalizer(abc.ABC):
    _memory: MemoryInterface

    def __init__(self, *, memory: MemoryInterface, verbose=False) -> None:
        self._memory = memory
        self.id = str(uuid4())

    def send_prompt(
        self,
        request: NormalizerRequest,
        target: PromptTarget,
        conversation_id: str = None,
        sequence: int = -1,
        labels={},
        orchestrator: "Orchestrator" = None,
    ) -> PromptRequestResponse:
        """
        Sends a single request to a target
        """

        request = self._get_prompt_request_response(
            request=request,
            target=target,
            conversation_id=conversation_id,
            sequence=sequence,
            labels=labels,
            orchestrator=orchestrator,
        )
        try:
            # Use the synchronous prompt sending method by default.

            return target.send_prompt(prompt_request=request)
        except NotImplementedError:
            # Alternatively, use async if sync is unavailable.
            pool = concurrent.futures.ThreadPoolExecutor()
            return pool.submit(asyncio.run, target.send_prompt_async(prompt_request=request)).result()

    async def send_prompt_async(
        self,
        request: NormalizerRequest,
        target: PromptTarget,
        conversation_id: str = None,
        sequence: int = -1,
        labels={},
        orchestrator: "Orchestrator" = None,
    ) -> PromptRequestResponse:
        """
        Sends a single request to a target
        """

        request = self._get_prompt_request_response(
            request=request,
            target=target,
            conversation_id=conversation_id,
            sequence=sequence,
            labels=labels,
            orchestrator=orchestrator,
        )

        response = await target.send_prompt_async(prompt_request=request)
        return response

    async def send_prompt_batch_to_target_async(
        self,
        requests: list[NormalizerRequest],
        target: PromptTarget,
        labels={},
        orchestrator: "Orchestrator" = None,
        batch_size: int = 10,
    ) -> list[PromptRequestResponse]:

        results = []

        for prompts_batch in self._chunked_prompts(requests, batch_size):
            tasks = []
            for prompt in prompts_batch:
                tasks.append(
                    self.send_prompt_async(request=prompt, target=target, labels=labels, orchestrator=orchestrator)
                )

            batch_results = await asyncio.gather(*tasks)
            results.extend(batch_results)

        return results

    def _chunked_prompts(self, prompts, size):
        for i in range(0, len(prompts), size):
            yield prompts[i : i + size]

    def _get_prompt_request_response(
        self,
        request: NormalizerRequest,
        target: PromptTarget,
        conversation_id: str = None,
        sequence: int = -1,
        labels={},
        orchestrator: "Orchestrator" = None,
    ) -> PromptRequestResponse:

        entries = []

        for request_piece in request.request_pieces:

            converted_prompt_text = request_piece.prompt_text
            for converter in request_piece.prompt_converters:
                converted_prompt_text = converter.convert(
                    prompt=converted_prompt_text, input_type=request_piece.prompt_data_type
                )

            entries.append(
                PromptRequestPiece(
                    role="user",
                    original_prompt_text=request_piece.prompt_text,
                    converted_prompt_text=converted_prompt_text,
                    conversation_id=conversation_id,
                    sequence=sequence,
                    labels=labels,
                    prompt_metadata=request_piece.metadata,
                    converters=request_piece.prompt_converters,
                    prompt_target=target,
                    orchestrator=orchestrator,
                    original_prompt_data_type=request_piece.prompt_data_type,
                    converted_prompt_data_type=request_piece.prompt_data_type,
                )
            )

        return PromptRequestResponse(request_pieces=entries)
