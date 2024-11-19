# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import abc
from typing import Optional
from uuid import uuid4

from pyrit.common.batch_helper import batch_task_async
from pyrit.memory import MemoryInterface, CentralMemory
from pyrit.models import PromptRequestResponse, PromptRequestPiece, PromptDataType, construct_response_from_request
from pyrit.prompt_converter import PromptConverter
from pyrit.prompt_target import PromptTarget

from pyrit.prompt_normalizer.normalizer_request import NormalizerRequest
from pyrit.prompt_normalizer.prompt_response_converter_configuration import PromptResponseConverterConfiguration


class PromptNormalizer(abc.ABC):
    _memory: MemoryInterface = None

    def __init__(self) -> None:
        self._memory = CentralMemory.get_memory_instance()
        self.id = str(uuid4())

    async def send_prompt_async(
        self,
        *,
        normalizer_request: NormalizerRequest,
        target: PromptTarget,
        sequence: int = -1,
        labels: Optional[dict[str, str]] = None,
        orchestrator_identifier: Optional[dict[str, str]] = None,
    ) -> PromptRequestResponse:
        """
        Sends a single request to a target.

        Args:
            normalizer_request (NormalizerRequest): The request to be sent.
            target (PromptTarget): The target to send the request to.
            sequence (int, Optional): The sequence number. Defaults to -1.
            labels (dict[str, str], Optional): Additional labels for the request. Defaults to None.
            orchestrator_identifier (dict[str, str], Optional): The orchestrator identifier. Defaults to None.

        Returns:
            PromptRequestResponse: The response received from the target.
        """

        request = await self._build_prompt_request_response(
            request=normalizer_request,
            target=target,
            sequence=sequence,
            labels=labels,
            orchestrator_identifier=orchestrator_identifier,
        )

        response = None

        try:
            response = await target.send_prompt_async(prompt_request=request)
            self._memory.add_request_response_to_memory(request=request)
        except Exception as ex:
            # Ensure request to memory before processing exception
            self._memory.add_request_response_to_memory(request=request)

            error_response = construct_response_from_request(
                request=request.request_pieces[0],
                response_text_pieces=[str(ex)],
                response_type="error",
                error="processing",
            )

            self._memory.add_request_response_to_memory(request=error_response)
            raise

        if response is None:
            return None

        await self.convert_response_values(
            response_converter_configurations=normalizer_request.response_converters, prompt_response=response
        )

        self._memory.add_request_response_to_memory(request=response)

        return response

    async def send_prompt_batch_to_target_async(
        self,
        *,
        requests: list[NormalizerRequest],
        target: PromptTarget,
        labels: Optional[dict[str, str]] = None,
        orchestrator_identifier: Optional[dict[str, str]] = None,
        batch_size: int = 10,
    ) -> list[PromptRequestResponse]:
        """
        Sends a batch of prompts to the target asynchronously.

        Args:
            requests (list[NormalizerRequest]): A list of NormalizerRequest objects representing the prompts to
                be sent.
            target (PromptTarget): The target to which the prompts should be sent.
            labels (dict[str, str], Optional): Additional labels to be included with the prompts. Defaults to None
            orchestrator_identifier (dict[str, str], Optional): The identifier of the orchestrator used for sending
                the prompts. Defaults to None.
            batch_size (int, Optional): The size of each batch of prompts. Defaults to 10.

        Returns:
            list[PromptRequestResponse]: A list of PromptRequestResponse objects representing the responses
                received for each prompt.
        """

        return await batch_task_async(
            prompt_target=target,
            batch_size=batch_size,
            items_to_batch=[requests],
            task_func=self.send_prompt_async,
            task_arguments=["normalizer_request"],
            target=target,
            labels=labels,
            orchestrator_identifier=orchestrator_identifier,
        )

    async def convert_response_values(
        self,
        response_converter_configurations: list[PromptResponseConverterConfiguration],
        prompt_response: PromptRequestResponse,
    ):

        for response_piece_index, response_piece in enumerate(prompt_response.request_pieces):
            for converter_configuration in response_converter_configurations:
                indexes = converter_configuration.indexes_to_apply
                data_types = converter_configuration.prompt_data_types_to_apply

                if indexes and response_piece_index not in indexes:
                    continue
                if data_types and response_piece.original_value_data_type not in data_types:
                    continue

                for converter in converter_configuration.converters:
                    converter_output = await converter.convert_async(
                        prompt=response_piece.original_value, input_type=response_piece.original_value_data_type
                    )
                    response_piece.converted_value = converter_output.output_text
                    response_piece.converted_value_data_type = converter_output.output_type

    async def _build_prompt_request_response(
        self,
        *,
        request: NormalizerRequest,
        target: PromptTarget,
        sequence: int = -1,
        labels: Optional[dict[str, str]] = None,
        orchestrator_identifier: Optional[dict[str, str]] = None,
    ) -> PromptRequestResponse:
        """
        Builds a prompt request response based on the given parameters.

        Applies parameters and converters to the prompt text and puts all the pieces together.

        Args:
            request (NormalizerRequest): The normalizer request object.
            target (PromptTarget): The prompt target object.
            sequence (int, Optional): The sequence number. Defaults to -1.
            labels (dict[str, str], Optional): The labels dictionary. Defaults to None.
            orchestrator_identifier (dict[str, str], Optional): The identifier of the orchestrator used for sending
                the prompts. Defaults to None.

        Returns:
            PromptRequestResponse: The prompt request response object.
        """

        entries = []

        # All prompt request pieces within PromptRequestResponse needs to have same conversation ID.
        conversation_id = request.conversation_id if request.conversation_id else str(uuid4())
        for request_piece in request.request_pieces:

            converted_prompt_text, converted_prompt_type = await self._get_converted_value_and_type(
                request_converters=request_piece.request_converters,
                prompt_value=request_piece.prompt_value,
                prompt_data_type=request_piece.prompt_data_type,
            )

            converter_identifiers = [converter.get_identifier() for converter in request_piece.request_converters]
            prompt_request_piece = PromptRequestPiece(
                role="user",
                original_value=request_piece.prompt_value,
                converted_value=converted_prompt_text,
                conversation_id=conversation_id,
                sequence=sequence,
                labels=labels,
                prompt_metadata=request_piece.metadata,
                converter_identifiers=converter_identifiers,
                prompt_target_identifier=target.get_identifier(),
                orchestrator_identifier=orchestrator_identifier,
                original_value_data_type=request_piece.prompt_data_type,
                converted_value_data_type=converted_prompt_type,
            )
            await prompt_request_piece.compute_sha256()
            entries.append(prompt_request_piece)

        return PromptRequestResponse(request_pieces=entries)

    async def _get_converted_value_and_type(
        self,
        request_converters: list[PromptConverter],
        prompt_value: str,
        prompt_data_type: PromptDataType,
    ):
        converted_prompt_value = prompt_value
        converted_prompt_type = prompt_data_type

        for converter in request_converters:
            converter_output = await converter.convert_tokens_async(
                prompt=converted_prompt_value, input_type=converted_prompt_type
            )
            converted_prompt_value = converter_output.output_text
            converted_prompt_type = converter_output.output_type

        return converted_prompt_value, converted_prompt_type
