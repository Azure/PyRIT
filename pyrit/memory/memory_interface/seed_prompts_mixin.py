# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Seed prompts mixin for MemoryInterface containing seed prompt-related operations."""

import logging
import uuid
from datetime import datetime
from typing import MutableSequence, Optional, Sequence, Union

from sqlalchemy import and_

from pyrit.memory.memory_models import SeedPromptEntry
from pyrit.models import DataTypeSerializer, SeedPrompt, SeedPromptDataset, SeedPromptGroup, data_serializer_factory

logger = logging.getLogger(__name__)


class MemorySeedPromptsMixin:
    """Mixin providing seed prompt-related methods for memory management."""

    def get_seed_prompts(
        self,
        *,
        value: Optional[str] = None,
        value_sha256: Optional[Sequence[str]] = None,
        dataset_name: Optional[str] = None,
        data_types: Optional[Sequence[str]] = None,
        harm_categories: Optional[Sequence[str]] = None,
        added_by: Optional[str] = None,
        authors: Optional[Sequence[str]] = None,
        groups: Optional[Sequence[str]] = None,
        source: Optional[str] = None,
        parameters: Optional[Sequence[str]] = None,
        metadata: Optional[dict[str, Union[str, int]]] = None,
    ) -> Sequence[SeedPrompt]:
        """
        Retrieves a list of seed prompts based on the specified filters.

        Args:
            value (str): The value to match by substring. If None, all values are returned.
            value_sha256 (str): The SHA256 hash of the value to match. If None, all values are returned.
            dataset_name (str): The dataset name to match. If None, all dataset names are considered.
            data_types (Optional[Sequence[str], Optional): List of data types to filter seed prompts by
                (e.g., text, image_path).
            harm_categories (Sequence[str]): A list of harm categories to filter by. If None,
            all harm categories are considered.
                Specifying multiple harm categories returns only prompts that are marked with all harm categories.
            added_by (str): The user who added the prompts.
            authors (Sequence[str]): A list of authors to filter by.
                Note that this filters by substring, so a query for "Adam Jones" may not return results if the record
                is "A. Jones", "Jones, Adam", etc. If None, all authors are considered.
            groups (Sequence[str]): A list of groups to filter by. If None, all groups are considered.
            source (str): The source to filter by. If None, all sources are considered.
            parameters (Sequence[str]): A list of parameters to filter by. Specifying parameters effectively returns
                prompt templates instead of prompts.

        Returns:
            Sequence[SeedPrompt]: A list of prompts matching the criteria.
        """
        conditions = []

        # Apply filters for non-list fields
        if value:
            conditions.append(SeedPromptEntry.value.contains(value))
        if value_sha256:
            conditions.append(SeedPromptEntry.value_sha256.in_(value_sha256))
        if dataset_name:
            conditions.append(SeedPromptEntry.dataset_name == dataset_name)
        if data_types:
            data_type_conditions = SeedPromptEntry.data_type.in_(data_types)
            conditions.append(data_type_conditions)
        if added_by:
            conditions.append(SeedPromptEntry.added_by == added_by)
        if source:
            conditions.append(SeedPromptEntry.source == source)

        self._add_list_conditions(field=SeedPromptEntry.harm_categories, values=harm_categories, conditions=conditions)
        self._add_list_conditions(field=SeedPromptEntry.authors, values=authors, conditions=conditions)
        self._add_list_conditions(field=SeedPromptEntry.groups, values=groups, conditions=conditions)

        if parameters:
            self._add_list_conditions(field=SeedPromptEntry.parameters, values=parameters, conditions=conditions)

        if metadata:
            conditions.append(self._get_seed_prompts_metadata_conditions(metadata=metadata))

        try:
            memory_entries: Sequence[SeedPromptEntry] = self._query_entries(
                SeedPromptEntry,
                conditions=and_(*conditions) if conditions else None,
            )  # type: ignore
            return [memory_entry.get_seed_prompt() for memory_entry in memory_entries]
        except Exception as e:
            logger.exception(f"Failed to retrieve prompts with dataset name {dataset_name} with error {e}")
            return []

    async def _serialize_seed_prompt_value(self, prompt: SeedPrompt) -> str:
        """
        Serializes the value of a seed prompt based on its data type.

        Args:
            prompt (SeedPrompt): The seed prompt to serialize. Must have a valid `data_type`.

        Returns:
            str: The serialized value for the prompt.

        Raises:
            ValueError: If the `data_type` of the prompt is unsupported.
        """
        extension = DataTypeSerializer.get_extension(prompt.value)
        if extension:
            extension = extension.lstrip(".")
        serializer = data_serializer_factory(
            category="seed-prompt-entries", data_type=prompt.data_type, value=prompt.value, extension=extension
        )
        serialized_prompt_value = None
        if prompt.data_type == "image_path":
            # Read the image
            original_img_bytes = await serializer.read_data_base64()
            # Save the image
            await serializer.save_b64_image(original_img_bytes)
            serialized_prompt_value = str(serializer.value)
        elif prompt.data_type in ["audio_path", "video_path"]:
            audio_bytes = await serializer.read_data()
            await serializer.save_data(data=audio_bytes)
            serialized_prompt_value = str(serializer.value)
        return serialized_prompt_value

    async def add_seed_prompts_to_memory_async(
        self, *, prompts: Sequence[SeedPrompt], added_by: Optional[str] = None
    ) -> None:
        """
        Inserts a list of prompts into the memory storage.

        Args:
            prompts (Sequence[SeedPrompt]): A list of prompts to insert.
            added_by (str): The user who added the prompts.
        """
        entries: MutableSequence[SeedPromptEntry] = []
        current_time = datetime.now()
        for prompt in prompts:
            if added_by:
                prompt.added_by = added_by
            if not prompt.added_by:
                raise ValueError(
                    """The 'added_by' attribute must be set for each prompt.
                    Set it explicitly or pass a value to the 'added_by' parameter."""
                )
            if prompt.date_added is None:
                prompt.date_added = current_time

            prompt.set_encoding_metadata()

            serialized_prompt_value = await self._serialize_seed_prompt_value(prompt)
            if serialized_prompt_value:
                prompt.value = serialized_prompt_value

            await prompt.set_sha256_value_async()

            if not self.get_seed_prompts(value_sha256=[prompt.value_sha256], dataset_name=prompt.dataset_name):
                entries.append(SeedPromptEntry(entry=prompt))

        self._insert_entries(entries=entries)

    def get_seed_prompt_dataset_names(self) -> Sequence[str]:
        """
        Returns a list of all seed prompt dataset names in the memory storage.
        """
        try:
            entries: Sequence[SeedPromptEntry] = self._query_entries(
                SeedPromptEntry,
                conditions=and_(
                    SeedPromptEntry.dataset_name is not None, SeedPromptEntry.dataset_name != ""  # type: ignore
                ),
                distinct=True,
            )
            # Extract unique dataset names from the entries
            dataset_names = set()
            for entry in entries:
                if entry.dataset_name:
                    dataset_names.add(entry.dataset_name)
            return list(dataset_names)
        except Exception as e:
            logger.exception(f"Failed to retrieve dataset names with error {e}")
            return []

    async def add_seed_prompt_groups_to_memory(
        self, *, prompt_groups: Sequence[SeedPromptGroup], added_by: Optional[str] = None
    ) -> None:
        """
        Inserts a list of seed prompt groups into the memory storage.

        Args:
            prompt_groups (Sequence[SeedPromptGroup]): A list of prompt groups to insert.
            added_by (str): The user who added the prompt groups.

        Raises:
            ValueError: If a prompt group does not have at least one prompt.
            ValueError: If prompt group IDs are inconsistent within the same prompt group.
        """
        if not prompt_groups:
            raise ValueError("At least one prompt group must be provided.")
        # Validates the prompt group IDs and sets them if possible before leveraging
        # the add_seed_prompts_to_memory method.
        all_prompts: MutableSequence[SeedPrompt] = []
        for prompt_group in prompt_groups:
            if not prompt_group.prompts:
                raise ValueError("Prompt group must have at least one prompt.")
            # Determine the prompt group ID.
            # It should either be set uniformly or generated if not set.
            # Inconsistent prompt group IDs will raise an error.
            group_id_set = set(prompt.prompt_group_id for prompt in prompt_group.prompts)
            if len(group_id_set) > 1:
                raise ValueError(
                    f"""Inconsistent 'prompt_group_id' attribute between members of the
                    same prompt group. Found {group_id_set}"""
                )
            prompt_group_id = group_id_set.pop() or uuid.uuid4()
            for prompt in prompt_group.prompts:
                prompt.prompt_group_id = prompt_group_id
            all_prompts.extend(prompt_group.prompts)
        await self.add_seed_prompts_to_memory_async(prompts=all_prompts, added_by=added_by)

    def get_seed_prompt_groups(
        self,
        *,
        value_sha256: Optional[Sequence[str]] = None,
        dataset_name: Optional[str] = None,
        data_types: Optional[Sequence[str]] = None,
        harm_categories: Optional[Sequence[str]] = None,
        added_by: Optional[str] = None,
        authors: Optional[Sequence[str]] = None,
        groups: Optional[Sequence[str]] = None,
        source: Optional[str] = None,
    ) -> Sequence[SeedPromptGroup]:
        """Retrieves groups of seed prompts based on the provided filtering criteria.

        Args:
            value_sha256 (Optional[Sequence[str]], Optional): SHA256 hash of value to filter seed prompt groups by.
            dataset_name (Optional[str], Optional): Name of the dataset to filter seed prompts.
            data_types (Optional[Sequence[str]], Optional): List of data types to filter seed prompts by
            (e.g., text, image_path).
            harm_categories (Optional[Sequence[str]], Optional): List of harm categories to filter seed prompts by.
            added_by (Optional[str], Optional): The user who added the seed prompt groups to filter by.
            authors (Optional[Sequence[str]], Optional): List of authors to filter seed prompt groups by.
            groups (Optional[Sequence[str]], Optional): List of groups to filter seed prompt groups by.
            source (Optional[str], Optional): The source from which the seed prompts originated.

        Returns:
            Sequence[SeedPromptGroup]: A list of `SeedPromptGroup` objects that match the filtering criteria.
        """
        seed_prompts = self.get_seed_prompts(
            value_sha256=value_sha256,
            dataset_name=dataset_name,
            data_types=data_types,
            harm_categories=harm_categories,
            added_by=added_by,
            authors=authors,
            groups=groups,
            source=source,
        )
        seed_prompt_groups = SeedPromptDataset.group_seed_prompts_by_prompt_group_id(seed_prompts)
        return seed_prompt_groups
