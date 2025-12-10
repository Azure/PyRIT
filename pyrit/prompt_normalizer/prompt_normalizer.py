# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import asyncio
import copy
import logging
import traceback
from typing import Any, List, Optional
from uuid import uuid4

from pyrit.exceptions import EmptyResponseException
from pyrit.memory import CentralMemory, MemoryInterface
from pyrit.models import (
    Message,
    MessagePiece,
    SeedGroup,
    construct_response_from_request,
)
from pyrit.prompt_normalizer import NormalizerRequest, PromptConverterConfiguration
from pyrit.prompt_target import PromptTarget
from pyrit.prompt_target.batch_helper import batch_task_async

logger = logging.getLogger(__name__)


class PromptNormalizer:
    """
    Handles normalization and processing of prompts before they are sent to targets.
    """

    _memory: MemoryInterface = None

    def __init__(self, start_token: str = "⟪", end_token: str = "⟫") -> None:
        """
        Initialize the PromptNormalizer.

        start_token and end_token are used to delineate which part of a prompt is converted.
        """
        self._memory = CentralMemory.get_memory_instance()
        self._start_token = start_token
        self._end_token = end_token
        self.id = str(uuid4())

    async def send_prompt_async(
        self,
        *,
        seed_group: SeedGroup,
        target: PromptTarget,
        conversation_id: Optional[str] = None,
        request_converter_configurations: list[PromptConverterConfiguration] = [],
        response_converter_configurations: list[PromptConverterConfiguration] = [],
        labels: Optional[dict[str, str]] = None,
        attack_identifier: Optional[dict[str, str]] = None,
    ) -> Message:
        """
        Send a single request to a target.

        Args:
            seed_group (SeedGroup): The seed group to be sent.
            target (PromptTarget): The target to which the prompt is sent.
            conversation_id (str, optional): The ID of the conversation. Defaults to None.
            request_converter_configurations (list[PromptConverterConfiguration], optional): Configurations for
                converting the request. Defaults to an empty list.
            response_converter_configurations (list[PromptConverterConfiguration], optional): Configurations for
                converting the response. Defaults to an empty list.
            labels (Optional[dict[str, str]], optional): Labels associated with the request. Defaults to None.
            attack_identifier (Optional[dict[str, str]], optional): Identifier for the attack. Defaults to
                None.

        Raises:
            Exception: If an error occurs during the request processing.
            ValueError: If the prompts in the SeedGroup are not part of the same sequence.

        Returns:
            Message: The response received from the target.
        """
        # Validates that the SeedPrompts in the SeedGroup are part of the same sequence
        if len(set(prompt.sequence for prompt in seed_group.prompts)) > 1:
            raise ValueError("All SeedPrompts in the SeedGroup must have the same sequence.")

        request = await self._build_message(
            seed_group=seed_group,
            conversation_id=conversation_id,
            request_converter_configurations=request_converter_configurations,
            target=target,
            labels=labels,
            attack_identifier=attack_identifier,
        )

        await self._calc_hash(request=request)

        responses = None

        try:
            responses = await target.send_prompt_async(message=request)
            self._memory.add_message_to_memory(request=request)
        except EmptyResponseException:
            # Empty responses are retried, but we don't want them to stop execution
            self._memory.add_message_to_memory(request=request)

            responses = [
                construct_response_from_request(
                    request=request.message_pieces[0],
                    response_text_pieces=[""],
                    response_type="text",
                    error="empty",
                )
            ]

        except Exception as ex:
            # Ensure request to memory before processing exception
            self._memory.add_message_to_memory(request=request)

            error_response = construct_response_from_request(
                request=request.message_pieces[0],
                response_text_pieces=[f"{ex}\n{repr(ex)}\n{traceback.format_exc()}"],
                response_type="error",
                error="processing",
            )

            await self._calc_hash(request=error_response)
            self._memory.add_message_to_memory(request=error_response)
            cid = request.message_pieces[0].conversation_id if request and request.message_pieces else None
            raise Exception(f"Error sending prompt with conversation ID: {cid}") from ex

        # handling empty responses message list and None responses
        if not responses or not any(responses):
            return None

        # Process all response messages (targets return list[Message])
        # Only apply response converters to the last message (final response)
        # Intermediate messages are tool calls/outputs that don't need conversion
        for i, resp in enumerate(responses):
            is_last = i == len(responses) - 1
            if is_last:
                await self.convert_values(converter_configurations=response_converter_configurations, message=resp)
            await self._calc_hash(request=resp)
            self._memory.add_message_to_memory(request=resp)

        # Return the last response for backward compatibility
        return responses[-1]

    async def send_prompt_batch_to_target_async(
        self,
        *,
        requests: list[NormalizerRequest],
        target: PromptTarget,
        labels: Optional[dict[str, str]] = None,
        attack_identifier: Optional[dict[str, str]] = None,
        batch_size: int = 10,
    ) -> list[Message]:
        """
        Send a batch of prompts to the target asynchronously.

        Args:
            requests (list[NormalizerRequest]): A list of NormalizerRequest objects to be sent.
            target (PromptTarget): The target to which the prompts are sent.
            labels (Optional[dict[str, str]], optional): A dictionary of labels to be included with the request.
                Defaults to None.
            attack_identifier (Optional[dict[str, str]], optional): A dictionary identifying the attack.
                Defaults to None.
            batch_size (int, optional): The number of prompts to include in each batch. Defaults to 10.

        Returns:
            list[Message]: A list of Message objects representing the responses
                received for each prompt.
        """
        batch_items: List[List[Any]] = [
            [request.seed_group for request in requests],
            [request.request_converter_configurations for request in requests],
            [request.response_converter_configurations for request in requests],
            [request.conversation_id for request in requests],
        ]

        batch_item_keys = [
            "seed_group",
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
            attack_identifier=attack_identifier,
        )

        # Filter out None responses (e.g., from empty responses)
        return [response for response in responses if response is not None]

    async def convert_values(
        self,
        converter_configurations: list[PromptConverterConfiguration],
        message: Message,
    ) -> None:
        """
        Apply converter configurations to message pieces.

        Args:
            converter_configurations (list[PromptConverterConfiguration]): List of configurations specifying
                which converters to apply and to which message pieces.
            message (Message): The message containing pieces to be converted.
        """
        for converter_configuration in converter_configurations:
            for piece_index, piece in enumerate(message.message_pieces):
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

    async def _calc_hash(self, request: Message) -> None:
        """Add a request to the memory."""
        tasks = [asyncio.create_task(piece.set_sha256_values_async()) for piece in request.message_pieces]
        await asyncio.gather(*tasks)

    async def _build_message(
        self,
        *,
        seed_group: SeedGroup,
        conversation_id: str,
        request_converter_configurations: list[PromptConverterConfiguration],
        target: PromptTarget,
        labels: dict[str, str],
        attack_identifier: Optional[dict[str, str]] = None,
    ) -> Message:
        """
        Build a message based on the given parameters.

        Applies parameters and converters to the prompt text and puts all the pieces together.

        Args:
            seed_group (SeedGroup): The group of seed prompts to be used.
            conversation_id (str): The ID of the conversation.
            request_converter_configurations (list[PromptConverterConfiguration]): List of configurations for
                request converters.
            target (PromptTarget): The target for the prompt.
            labels (dict[str, str]): A dictionary of labels associated with the prompt.
            attack_identifier (Optional[dict[str, str]]): An optional dictionary for attack identifiers.

        Returns:
            Message: The message object.
        """
        entries = []

        # All message pieces within Message needs to have same conversation ID.
        conversation_id = conversation_id if conversation_id else str(uuid4())
        for seed_prompt in seed_group.prompts:
            message_piece = MessagePiece(
                role=seed_prompt.role,
                original_value=seed_prompt.value,
                conversation_id=conversation_id,
                sequence=seed_prompt.sequence,
                labels=labels,
                prompt_metadata=seed_prompt.metadata,
                prompt_target_identifier=target.get_identifier(),
                attack_identifier=attack_identifier,
                original_value_data_type=seed_prompt.data_type,
            )

            entries.append(message_piece)

        response = Message(message_pieces=entries)

        await self.convert_values(converter_configurations=request_converter_configurations, message=response)
        return response

    async def add_prepended_conversation_to_memory(
        self,
        conversation_id: str,
        should_convert: bool = True,
        converter_configurations: Optional[list[PromptConverterConfiguration]] = None,
        attack_identifier: Optional[dict[str, str]] = None,
        prepended_conversation: Optional[list[Message]] = None,
    ) -> Optional[list[Message]]:
        """
        Process the prepended conversation by converting it if needed and adding it to memory.

        Args:
            conversation_id (str): The conversation ID to use for the message pieces
            should_convert (bool): Whether to convert the prepended conversation
            converter_configurations (Optional[list[PromptConverterConfiguration]]): Configurations for converting the
                request
            attack_identifier (Optional[dict[str, str]]): Identifier for the attack
            prepended_conversation (Optional[list[Message]]): The conversation to prepend

        Returns:
            Optional[list[Message]]: The processed prepended conversation
        """
        if not prepended_conversation:
            return None

        # Create a deep copy of the prepended conversation to avoid modifying the original
        prepended_conversation = copy.deepcopy(prepended_conversation)

        for request in prepended_conversation:
            if should_convert and converter_configurations:
                await self.convert_values(message=request, converter_configurations=converter_configurations)
            for piece in request.message_pieces:
                piece.conversation_id = conversation_id
                if attack_identifier:
                    piece.attack_identifier = attack_identifier

                # if the piece is retrieved from somewhere else, it needs to be unique
                # and if not, this won't hurt anything
                piece.id = uuid4()

            self._memory.add_message_to_memory(request=request)

        return prepended_conversation
