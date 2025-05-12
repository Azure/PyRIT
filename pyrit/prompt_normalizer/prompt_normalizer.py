# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import abc
import asyncio
import copy
import logging
import traceback
from typing import Any, List, Optional
from uuid import uuid4

from pyrit.exceptions import EmptyResponseException
from pyrit.memory import CentralMemory, MemoryInterface
from pyrit.models import (
    PromptRequestPiece,
    PromptRequestResponse,
    construct_response_from_request,
)
from pyrit.models.filter_criteria import PromptConverterState, PromptFilterCriteria
from pyrit.models.seed_prompt import SeedPromptGroup
from pyrit.prompt_normalizer import NormalizerRequest, PromptConverterConfiguration
from pyrit.prompt_target import PromptTarget
from pyrit.prompt_target.batch_helper import batch_task_async

logger = logging.getLogger(__name__)


class PromptNormalizer(abc.ABC):
    _memory: MemoryInterface = None

    def __init__(self, start_token: str = "⟪", end_token: str = "⟫") -> None:
        """
        Initializes the PromptNormalizer.

        start_token and end_token are used to delineate which part of a prompt is converted.
        """
        self._memory = CentralMemory.get_memory_instance()
        self._start_token = start_token
        self._end_token = end_token
        self.id = str(uuid4())
        self._skip_criteria: Optional[PromptFilterCriteria] = None

    async def send_prompt_async(
        self,
        *,
        seed_prompt_group: SeedPromptGroup,
        target: PromptTarget,
        conversation_id: str = None,
        request_converter_configurations: list[PromptConverterConfiguration] = [],
        response_converter_configurations: list[PromptConverterConfiguration] = [],
        sequence: int = -1,
        labels: Optional[dict[str, str]] = None,
        orchestrator_identifier: Optional[dict[str, str]] = None,
    ) -> PromptRequestResponse:
        """
        Sends a single request to a target.

        Args:
            seed_prompt_group (SeedPromptGroup): The seed prompt group to be sent.
            target (PromptTarget): The target to which the prompt is sent.
            conversation_id (str, optional): The ID of the conversation. Defaults to None.
            request_converter_configurations (list[PromptConverterConfiguration], optional): Configurations for
                converting the request. Defaults to an empty list.
            response_converter_configurations (list[PromptConverterConfiguration], optional): Configurations for
                converting the response. Defaults to an empty list.
            sequence (int, optional): The sequence number of the request. Defaults to -1.
            labels (Optional[dict[str, str]], optional): Labels associated with the request. Defaults to None.
            orchestrator_identifier (Optional[dict[str, str]], optional): Identifier for the orchestrator. Defaults to
                None.

            Raises:
            Exception: If an error occurs during the request processing.

        Returns:
            PromptRequestResponse: The response received from the target.
        """

        request = await self._build_prompt_request_response(
            seed_prompt_group=seed_prompt_group,
            conversation_id=conversation_id,
            request_converter_configurations=request_converter_configurations,
            target=target,
            sequence=sequence,
            labels=labels,
            orchestrator_identifier=orchestrator_identifier,
        )

        await self._calc_hash(request=request)

        if self._should_skip_based_on_skip_criteria(request):
            return None

        response = None

        try:
            response = await target.send_prompt_async(prompt_request=request)
            self._memory.add_request_response_to_memory(request=request)
        except EmptyResponseException:
            # Empty responses are retried, but we don't want them to stop execution
            self._memory.add_request_response_to_memory(request=request)

            response = construct_response_from_request(
                request=request.request_pieces[0],
                response_text_pieces=[""],
                response_type="text",
                error="empty",
            )

        except Exception as ex:
            # Ensure request to memory before processing exception
            self._memory.add_request_response_to_memory(request=request)

            error_response = construct_response_from_request(
                request=request.request_pieces[0],
                response_text_pieces=[f"{ex}\n{repr(ex)}\n{traceback.format_exc()}"],
                response_type="error",
                error="processing",
            )

            await self._calc_hash(request=error_response)
            self._memory.add_request_response_to_memory(request=error_response)
            cid = request.request_pieces[0].conversation_id if request and request.request_pieces else None
            raise Exception(f"Error sending prompt with conversation ID: {cid}") from ex

        if response is None:
            return None

        await self.convert_values(converter_configurations=response_converter_configurations, request_response=response)

        await self._calc_hash(request=response)
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
            requests (list[NormalizerRequest]): A list of NormalizerRequest objects to be sent.
            target (PromptTarget): The target to which the prompts are sent.
            labels (Optional[dict[str, str]], optional): A dictionary of labels to be included with the request.
                Defaults to None.
            orchestrator_identifier (Optional[dict[str, str]], optional): A dictionary identifying the orchestrator.
                Defaults to None.
            batch_size (int, optional): The number of prompts to include in each batch. Defaults to 10.

        Returns:
            list[PromptRequestResponse]: A list of PromptRequestResponse objects representing the responses
                received for each prompt.
        """

        batch_items: List[List[Any]] = [
            [request.seed_prompt_group for request in requests],
            [request.request_converter_configurations for request in requests],
            [request.response_converter_configurations for request in requests],
            [request.conversation_id for request in requests],
        ]

        batch_item_keys = [
            "seed_prompt_group",
            "request_converter_configurations",
            "response_converter_configurations",
            "conversation_id",
        ]

        responses = await batch_task_async(
            prompt_target=target,
            batch_size=batch_size,
            items_to_batch=batch_items,
            task_func=self.send_prompt_async,
            task_arguments=batch_item_keys,
            target=target,
            labels=labels,
            orchestrator_identifier=orchestrator_identifier,
        )

        # send_prompt_async can return None if the prompt is skipped
        return [response for response in responses if response is not None]

    async def convert_values(
        self,
        converter_configurations: list[PromptConverterConfiguration],
        request_response: PromptRequestResponse,
    ):

        for converter_configuration in converter_configurations:
            for piece_index, piece in enumerate(request_response.request_pieces):
                indexes = converter_configuration.indexes_to_apply
                data_types = converter_configuration.prompt_data_types_to_apply

                if indexes and piece_index not in indexes:
                    continue
                if data_types and piece.converted_value_data_type not in data_types:
                    continue

                piece.converter_identifiers.extend(
                    [converter.get_identifier() for converter in converter_configuration.converters]
                )

                converted_text = piece.converted_value
                converted_text_data_type = piece.converted_value_data_type

                for converter in converter_configuration.converters:
                    converter_result = await converter.convert_tokens_async(
                        prompt=converted_text,
                        input_type=converted_text_data_type,
                        start_token=self._start_token,
                        end_token=self._end_token,
                    )
                    converted_text = converter_result.output_text
                    converted_text_data_type = converter_result.output_type

                piece.converted_value = converted_text
                piece.converted_value_data_type = converted_text_data_type

    def set_skip_criteria(self, skip_criteria: PromptFilterCriteria, skip_value_type: PromptConverterState) -> None:
        """
        Sets the skip criteria for the orchestrator.

        If prompts match this in memory and are the same as one being sent, then they won't be sent to a target.

        Prompts are the same if either the original prompt or the converted prompt, determined by skip_value_type flag.
        """
        self._skip_criteria = skip_criteria

        prompts_to_skip = self._memory.get_prompt_request_pieces(
            role="user",
            orchestrator_id=self._skip_criteria.orchestrator_id,
            conversation_id=self._skip_criteria.conversation_id,
            prompt_ids=self._skip_criteria.prompt_ids,
            labels=self._skip_criteria.labels,
            sent_after=self._skip_criteria.sent_after,
            sent_before=self._skip_criteria.sent_before,
            original_values=self._skip_criteria.original_values,
            converted_values=self._skip_criteria.converted_values,
            data_type=self._skip_criteria.data_type,
            not_data_type=self._skip_criteria.not_data_type,
            converted_value_sha256=self._skip_criteria.converted_value_sha256,
        )

        self._original_sha256_prompts_to_skip = [
            prompt.original_value_sha256 for prompt in prompts_to_skip if prompt.original_value_sha256
        ]

        self._converted_sha256_prompts_to_skip = [
            prompt.converted_value_sha256 for prompt in prompts_to_skip if prompt.converted_value_sha256
        ]

        self._skip_value_type = skip_value_type

    def _should_skip_based_on_skip_criteria(self, prompt_request: PromptRequestResponse) -> bool:
        """
        Filters out prompts from prompt_request_list that match the skip criteria.

        Every request_piece of the prompt_request needs to have matching sha256 to skip.
        """
        if not self._skip_criteria:
            return False

        for user_prompt in prompt_request.request_pieces:
            if self._skip_value_type == "converted":
                if user_prompt.converted_value_sha256 not in self._converted_sha256_prompts_to_skip:
                    return False
            else:
                if user_prompt.original_value_sha256 not in self._original_sha256_prompts_to_skip:
                    return False
        return True

    async def _calc_hash(self, request: PromptRequestResponse) -> None:
        """
        Adds a request to the memory.
        """
        tasks = [asyncio.create_task(piece.set_sha256_values_async()) for piece in request.request_pieces]
        await asyncio.gather(*tasks)

    async def _build_prompt_request_response(
        self,
        *,
        seed_prompt_group: SeedPromptGroup,
        conversation_id: str,
        request_converter_configurations: list[PromptConverterConfiguration],
        target: PromptTarget,
        sequence: int,
        labels: dict[str, str],
        orchestrator_identifier: Optional[dict[str, str]],
    ) -> PromptRequestResponse:
        """
        Builds a prompt request response based on the given parameters.

        Applies parameters and converters to the prompt text and puts all the pieces together.

        Args:
            seed_prompt_group (SeedPromptGroup): The group of seed prompts to be used.
            conversation_id (str): The ID of the conversation.
            request_converter_configurations (list[PromptConverterConfiguration]): List of configurations for
                request converters.
            target (PromptTarget): The target for the prompt.
            sequence (int): The sequence number of the prompt.
            labels (dict[str, str]): A dictionary of labels associated with the prompt.
            orchestrator_identifier (Optional[dict[str, str]]): An optional dictionary for orchestrator identifiers.

        Returns:
            PromptRequestResponse: The prompt request response object.
        """

        entries = []

        # All prompt request pieces within PromptRequestResponse needs to have same conversation ID.
        conversation_id = conversation_id if conversation_id else str(uuid4())
        for seed_prompt in seed_prompt_group.prompts:

            prompt_request_piece = PromptRequestPiece(
                role="user",
                original_value=seed_prompt.value,
                conversation_id=conversation_id,
                sequence=sequence,
                labels=labels,
                prompt_metadata=seed_prompt.metadata,
                prompt_target_identifier=target.get_identifier(),
                orchestrator_identifier=orchestrator_identifier,
                original_value_data_type=seed_prompt.data_type,
            )

            entries.append(prompt_request_piece)

        response = PromptRequestResponse(request_pieces=entries)

        await self.convert_values(converter_configurations=request_converter_configurations, request_response=response)
        return response

    async def add_prepended_conversation_to_memory(
        self,
        prepended_conversation: Optional[list[PromptRequestResponse]],
        conversation_id: str,
        should_convert: bool = True,
        converter_configurations: Optional[list[PromptConverterConfiguration]] = None,
        orchestrator_identifier: Optional[dict[str, str]] = None,
    ) -> Optional[list[PromptRequestResponse]]:
        """
        Processes the prepended conversation by converting it if needed and adding it to memory.

        Args:
            prepended_conversation (Optional[list[PromptRequestResponse]]): The conversation to prepend
            conversation_id (str): The conversation ID to use for the request pieces
            should_convert (bool): Whether to convert the prepended conversation
            converter_configurations (Optional[list[PromptConverterConfiguration]]): Configurations for converting the
                request
            orchestrator_identifier (Optional[dict[str, str]]): Identifier for the orchestrator

        Returns:
            Optional[list[PromptRequestResponse]]: The processed prepended conversation
        """
        if not prepended_conversation:
            return None

        # Create a deep copy of the prepended conversation to avoid modifying the original
        prepended_conversation = copy.deepcopy(prepended_conversation)

        for request in prepended_conversation:
            if should_convert and converter_configurations:
                await self.convert_values(request_response=request, converter_configurations=converter_configurations)
            for piece in request.request_pieces:
                piece.conversation_id = conversation_id
                if orchestrator_identifier:
                    piece.orchestrator_identifier = orchestrator_identifier

                # if the piece is retrieved from somewhere else, it needs to be unique
                # and if not, this won't hurt anything
                piece.id = uuid4()

            self._memory.add_request_response_to_memory(request=request)

        return prepended_conversation
