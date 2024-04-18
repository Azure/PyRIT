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

    def __init__(self, *, memory: MemoryInterface) -> None:
        self._memory = memory
        self.id = str(uuid4())

    def send_prompt(
        self,
        normalizer_request: NormalizerRequest,
        target: PromptTarget,
        conversation_id: str = None,
        sequence: int = -1,
        labels={},
        orchestrator_identifier: dict[str, str] = None,
    ) -> PromptRequestResponse:
        """
        Sends a single request to a target.

        Args:
            normalizer_request (NormalizerRequest): The request to be sent.
            target (PromptTarget): The target to send the request to.
            conversation_id (str, optional): The ID of the conversation. Defaults to None.
            sequence (int, optional): The sequence number of the request. Defaults to -1.
            labels (dict, optional): Additional labels for the request. Defaults to {}.
            orchestrator_identifier (Orchestrator, optional): The orchestrator to use. Defaults to None.

        Returns:
            PromptRequestResponse: The response received from the target.
        """

        request_response = self._build_prompt_request_response(
            request=normalizer_request,
            target=target,
            conversation_id=conversation_id,
            sequence=sequence,
            labels=labels,
            orchestrator_identifier=orchestrator_identifier,
        )
        try:
            # Use the synchronous prompt sending method by default.
            return target.send_prompt(prompt_request=request_response)
        except NotImplementedError:
            # Alternatively, use async if sync is unavailable.
            pool = concurrent.futures.ThreadPoolExecutor()
            return pool.submit(asyncio.run, target.send_prompt_async(prompt_request=request_response)).result()

    async def send_prompt_async(
        self,
        normalizer_request: NormalizerRequest,
        target: PromptTarget,
        conversation_id: str = None,
        sequence: int = -1,
        labels=None,
        orchestrator_identifier: dict[str, str] = None,
    ) -> PromptRequestResponse:
        """
        Sends a single request to a target.

        Args:
            normalizer_request (NormalizerRequest): The request to be sent.
            target (PromptTarget): The target to send the request to.
            conversation_id (str, optional): The ID of the conversation. Defaults to None.
            sequence (int, optional): The sequence number. Defaults to -1.
            labels (dict, optional): Additional labels for the request. Defaults to None.
            orchestrator (Orchestrator, optional): The orchestrator. Defaults to None.

        Returns:
            PromptRequestResponse: The response received from the target.
        """
        request = self._build_prompt_request_response(
            request=normalizer_request,
            target=target,
            conversation_id=conversation_id,
            sequence=sequence,
            labels=labels,
            orchestrator_identifier=orchestrator_identifier,
        )

        response = await target.send_prompt_async(prompt_request=request)
        return response

    async def send_prompt_batch_to_target_async(
        self,
        requests: list[NormalizerRequest],
        target: PromptTarget,
        labels=None,
        orchestrator_identifier: dict[str, str] = None,
        batch_size: int = 10,
    ) -> list[PromptRequestResponse]:
        """
        Sends a batch of prompts to the target asynchronously.

        Args:
            requests (list[NormalizerRequest]): A list of NormalizerRequest objects representing the prompts to
                be sent.
            target (PromptTarget): The target to which the prompts should be sent.
            labels (dict, optional): Additional labels to be included with the prompts. Defaults to None
            orchestrator (Orchestrator, optional): The orchestrator to use for sending the prompts. Defaults
                to None.
            batch_size (int, optional): The size of each batch of prompts. Defaults to 10.

        Returns:
            list[PromptRequestResponse]: A list of PromptRequestResponse objects representing the responses
                received for each prompt.
        """

        results = []

        for prompts_batch in self._chunked_prompts(requests, batch_size):
            tasks = []
            for prompt in prompts_batch:
                tasks.append(
                    self.send_prompt_async(
                        normalizer_request=prompt,
                        target=target,
                        labels=labels,
                        orchestrator_identifier=orchestrator_identifier,
                    )
                )

            batch_results = await asyncio.gather(*tasks)
            results.extend(batch_results)

        return results

    def _chunked_prompts(self, prompts, size):
        for i in range(0, len(prompts), size):
            yield prompts[i : i + size]

    def _build_prompt_request_response(
        self,
        request: NormalizerRequest,
        target: PromptTarget,
        conversation_id: str = None,
        sequence: int = -1,
        labels=None,
        orchestrator_identifier: dict[str, str] = None,
    ) -> PromptRequestResponse:
        """
        Builds a prompt request response based on the given parameters.

        Applies parameters and converters to the prompt text and puts all the pieces together.

        Args:
            request (NormalizerRequest): The normalizer request object.
            target (PromptTarget): The prompt target object.
            conversation_id (str, optional): The conversation ID. Defaults to None.
            sequence (int, optional): The sequence number. Defaults to -1.
            labels (dict, optional): The labels dictionary. Defaults to None.
            orchestrator_identifier (Orchestrator, optional): The orchestrator object. Defaults to None.

        Returns:
            PromptRequestResponse: The prompt request response object.
        """

        entries = []

        for request_piece in request.request_pieces:

            converted_prompt_text = request_piece.prompt_text
            for converter in request_piece.prompt_converters:
                converted_prompt_text = converter.convert(
                    prompt=converted_prompt_text, input_type=request_piece.prompt_data_type
                )

            converter_identifiers = [converter.get_identifier() for converter in request_piece.prompt_converters]

            entries.append(
                PromptRequestPiece(
                    role="user",
                    original_prompt_text=request_piece.prompt_text,
                    converted_prompt_text=converted_prompt_text,
                    conversation_id=conversation_id,
                    sequence=sequence,
                    labels=labels,
                    prompt_metadata=request_piece.metadata,
                    converter_identifiers=converter_identifiers,
                    prompt_target_identifier=target.get_identifier(),
                    orchestrator_identifier=orchestrator_identifier,
                    original_prompt_data_type=request_piece.prompt_data_type,
                    converted_prompt_data_type=request_piece.prompt_data_type,
                )
            )

        return PromptRequestResponse(request_pieces=entries)
