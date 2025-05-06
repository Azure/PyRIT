# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import abc
import asyncio
from dataclasses import dataclass
import logging
import traceback
from typing import Any, Dict, List, Optional, Union
from uuid import uuid4
import uuid

from pyrit.exceptions import EmptyResponseException
from pyrit.memory import CentralMemory, MemoryInterface
from pyrit.models import (
    PromptRequestPiece,
    PromptRequestResponse,
    construct_response_from_request,
)
from pyrit.models.filter_criteria import PromptConverterState, PromptFilterCriteria
from pyrit.models.literals import PromptDataType
from pyrit.models.seed_prompt import SeedPrompt, SeedPromptGroup
from pyrit.prompt_converter.prompt_converter import PromptConverter
from pyrit.prompt_normalizer import NormalizerRequest, PromptConverterConfiguration
from pyrit.prompt_target import PromptTarget
from pyrit.prompt_target.batch_helper import batch_task_async

logger = logging.getLogger(__name__)

@dataclass
class PromptValidationCriteria:
    """Criteria for validating normalizer requests"""
    multipart_allowed: bool = False
    allowed_data_types: Optional[List[PromptDataType]] = None
    
    def __post_init__(self):
        if self.allowed_data_types is None:
            self.allowed_data_types = ["text"]

class PromptNormalizer(abc.ABC):

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
        conversation_id: str,
        request_converter_configurations: list[PromptConverterConfiguration] = [],
        response_converter_configurations: list[PromptConverterConfiguration] = [],
        sequence: int = -1,
        labels: Optional[dict[str, str]] = None,
        orchestrator_identifier: Optional[dict[str, str]] = None,
    ) -> Optional[PromptRequestResponse]:
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
            labels=labels or {},
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

    @staticmethod
    def build_normalizer_requests(
        *,
        prompts: List[str],
        prompt_type: PromptDataType = "text",
        converters: Optional[List[PromptConverter]] = None,
        metadata: Optional[dict[str, Union[str, int]]] = None,
    ) -> List[NormalizerRequest]:
        """
        Build normalizer requests from the provided prompts.
        
        Args:
            prompts: The list of prompts to normalize
            batch_size: The maximum batch size for sending prompts
            memory_labels: Optional memory labels for the attack
            
        Returns:
            A list of NormalizerRequest objects
        """
        if not prompts:
            raise ValueError("No prompts provided")

        return [
            PromptNormalizer._create_normalizer_request(
                prompt_text=prompt,
                prompt_type=prompt_type,
                converters=converters or [],
                metadata=metadata,
                conversation_id=str(uuid.uuid4()),
            )
            for prompt in prompts
        ]
    
    @staticmethod
    def _create_normalizer_request(
        *,
        prompt_text: str,
        prompt_type: PromptDataType = "text",
        conversation_id: Optional[str] = None,
        converters: Optional[List[PromptConverter]] = None,
        metadata: Optional[Dict[str, Union[str, int]]] = None,
    ) -> NormalizerRequest:
        """
        Create a normalizer request for a prompt.
        
        Args:
            prompt_text: The text of the prompt
            prompt_type: The type of the prompt data
            converters: Optional list of prompt converters
            metadata: Optional metadata for the prompt

        Returns:
            A NormalizerRequest object
        """
        seed_prompt_group = SeedPromptGroup(
            prompts=[
                SeedPrompt(
                    value=prompt_text,
                    data_type=prompt_type,
                    metadata=metadata,
                )
            ]
        )

        converter_configurations = [PromptConverterConfiguration(
            converters=converters or []
        )]

        return NormalizerRequest(
            seed_prompt_group=seed_prompt_group,
            request_converter_configurations=converter_configurations,
            conversation_id=conversation_id or str(uuid.uuid4()),
        )
    
    @staticmethod
    def validate_normalizer_requests(
        *,
        requests: List[NormalizerRequest],
        criteria: PromptValidationCriteria,
    ) -> None:
        """
        Validate normalizer requests based on provided criteria.
        
        Args:
            requests: The list of normalizer requests to validate
            criteria: The validation criteria
            
        Raises:
            ValueError: If any request doesn't meet the criteria
        """
        if not requests:
            raise ValueError("No normalizer requests provided")

        for request in requests:
            # Check if multipart messages are allowed
            if not criteria.multipart_allowed and request.is_multipart():
                raise ValueError("Multi-part messages not supported")
                
            # Check that data types are supported
            for prompt in request.seed_prompt_group.prompts:
                if criteria.allowed_data_types is not None and prompt.data_type not in criteria.allowed_data_types:
                    supported = ", ".join(criteria.allowed_data_types)
                    raise ValueError(f"Unsupported data type: {prompt.data_type}. Supported types: {supported}")