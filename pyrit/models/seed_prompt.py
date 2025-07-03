# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

import logging
import os
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Union

from jinja2 import BaseLoader, Environment, StrictUndefined, Template, Undefined
from pydantic.types import PositiveInt
from tinytag import TinyTag

from pyrit.common import utils
from pyrit.common.path import (
    DATASETS_PATH,
    DB_DATA_PATH,
    DOCS_CODE_PATH,
    HOME_PATH,
    LOG_PATH,
    PYRIT_PATH,
)
from pyrit.common.yaml_loadable import YamlLoadable
from pyrit.models import DataTypeSerializer
from pyrit.models.literals import ChatMessageRole, PromptDataType

logger = logging.getLogger(__name__)


class PartialUndefined(Undefined):
    # Return the original placeholder format
    def __str__(self):
        return f"{{{{ {self._undefined_name} }}}}" if self._undefined_name else ""

    def __repr__(self):
        return f"{{{{ {self._undefined_name} }}}}" if self._undefined_name else ""

    def __iter__(self):
        """Prevent Jinja from evaluating loops by returning a placeholder string instead of an iterable."""
        return self

    def __bool__(self):
        return True  # Ensures it doesn't evaluate to False


@dataclass
class SeedPrompt(YamlLoadable):
    """Represents a seed prompt with various attributes and metadata.

    value (str): The prompt text, which can include Jinja2 template parameters.
    value_sha256 (Optional[str]): The SHA256 hash of the prompt value.
    data_type (Optional[PromptDataType]): The type of data the prompt represents (e.g., text, image, audio).
    id (Optional[uuid.UUID]): Unique identifier for the prompt.
    name (Optional[str]): Name of the prompt.
    dataset_name (Optional[str]): Name of the dataset this prompt belongs to.
    harm_categories (Optional[Sequence[str]]): Categories of potential harm associated with the prompt.
    description (Optional[str]): Description of the prompt.
    authors (Optional[Sequence[str]]): Authors of the prompt.
    groups (Optional[Sequence[str]]): Groups associated with the prompt.
    source (Optional[str]): Source of the prompt.
    date_added (Optional[datetime]): Date when the prompt was added.
    added_by (Optional[str]): User who added the prompt.
    metadata (Optional[Dict[str, Union[str, int]]]): Additional metadata for the prompt
    parameters (Optional[Sequence[str]]): Parameters that can be used in the prompt.
    prompt_group_id (Optional[uuid.UUID]): a unique identifier generated for the prompt group this prompt
        belongs to.
    prompt_group_alias (Optional[str]): user set alias for the prompt group. Prompts in the same group will be
        sent together. If the group alias is not set, the prompt is assumed to be the only prompt in the group.
    sequence (Optional[int]): Sequence number for ordering prompts with the same seed alias. Relevant only if the prompt
        seed alias is set and there is more than one prompt in the prompt associated with that alias. Defaults to 0.
    role: (ChatMessageRole): The role of the prompt in a chat context--"system", "user", or "assistant". Defaults
        to "user".
    prompt_seed_alias (Optional[str]): An alias for the seed prompt, used to identify which seed prompts should be
        sent in the same turn. This is useful for multimodal prompts where multiple prompts need to be sent
        together in a single request. If not set, the prompt is assumed to be a single prompt that does not
        require grouping with other prompts.
    prompt_group_alias_id (Optional[uuid.UUID]): A unique identifier for the prompt group alias.
    """

    value: str
    value_sha256: Optional[str] = None
    data_type: Optional[PromptDataType] = None
    id: Optional[uuid.UUID] = field(default_factory=lambda: uuid.uuid4())
    name: Optional[str] = None
    dataset_name: Optional[str] = None
    harm_categories: Optional[Sequence[str]] = field(default_factory=lambda: [])
    description: Optional[str] = None
    authors: Optional[Sequence[str]] = field(default_factory=lambda: [])
    groups: Optional[Sequence[str]] = field(default_factory=lambda: [])
    source: Optional[str] = None
    date_added: Optional[datetime] = field(default_factory=lambda: datetime.now())
    added_by: Optional[str] = None
    metadata: Optional[Dict[str, Union[str, int]]] = field(default_factory=lambda: {})
    parameters: Optional[Sequence[str]] = field(default_factory=lambda: [])
    prompt_group_id: Optional[uuid.UUID] = None
    prompt_group_alias: Optional[str] = None
    sequence: Optional[int] = 0
    role: ChatMessageRole = "user"
    prompt_seed_alias: Optional[str] = None
    prompt_seed_id: Optional[uuid.UUID] = None

    TEMPLATE_PATHS = {
        "datasets_path": DATASETS_PATH,
        "pyrit_home_path": HOME_PATH,
        "pyrit_path": PYRIT_PATH,
        "db_data_path": DB_DATA_PATH,
        "log_path": LOG_PATH,
        "docs_code_path": DOCS_CODE_PATH,
    }

    def __post_init__(self) -> None:
        """Post-initialization to render the template to replace existing values"""
        self.value = self.render_template_value_silent(**self.TEMPLATE_PATHS)

        if self.role not in ChatMessageRole.__args__:  # type: ignore
            raise ValueError(f"Role {self.role} is not a valid role.")

        if not self.data_type:
            # If data_type is not provided, infer it from the value
            # Note: Does not assign 'error' or 'url' implicitly
            if os.path.isfile(self.value):
                _, ext = os.path.splitext(self.value)
                ext = ext.lstrip(".")
                if ext in ["mp4", "avi", "mov", "mkv", "ogv", "flv", "wmv", "webm"]:
                    self.data_type = "video_path"
                elif ext in ["flac", "mp3", "mpeg", "mpga", "m4a", "ogg", "wav"]:
                    self.data_type = "audio_path"
                elif ext in ["jpg", "jpeg", "png", "gif", "bmp", "tiff", "tif"]:
                    self.data_type = "image_path"
                else:
                    raise ValueError(f"Unable to infer data_type from file extension: {ext}")
            else:
                self.data_type = "text"

    def render_template_value(self, **kwargs) -> str:
        """Renders self.value as a template, applying provided parameters in kwargs

        Args:
            kwargs:Key-value pairs to replace in the SeedPrompt value.

        Returns:
            A new prompt with the parameters applied.

        Raises:
            ValueError: If parameters are missing or invalid in the template.
        """

        jinja_template = Template(self.value, undefined=StrictUndefined)

        try:
            return jinja_template.render(**kwargs)
        except Exception as e:
            raise ValueError(f"Error applying parameters: {str(e)}")

    def render_template_value_silent(self, **kwargs) -> str:
        """Renders self.value as a template, applying provided parameters in kwargs. For parameters in the template
         that are not provided as kwargs here, this function will leave them as is instead of raising an error.

        Args:
            kwargs: Key-value pairs to replace in the SeedPrompt value.

        Returns:
            A new prompt with the parameters applied.

        Raises:
            ValueError: If parameters are missing or invalid in the template.
        """
        # Create a Jinja template with PartialUndefined placeholders
        env = Environment(loader=BaseLoader, undefined=PartialUndefined)  # type: ignore
        jinja_template = env.from_string(self.value)

        try:
            # Render the template with the provided kwargs
            return jinja_template.render(**kwargs)
        except Exception as e:
            logging.error("Error rendering template: %s", e)
            return self.value

    async def set_sha256_value_async(self):
        """
        This method computes the SHA256 hash value asynchronously.
        It should be called after prompt `value` is serialized to text,
        as file paths used in the `value` may have changed from local to memory storage paths.

        Note, this method is async due to the blob retrieval. And because of that, we opted
        to take it out of main and setter functions. The disadvantage is that it must be explicitly called.
        """
        from pyrit.models.data_type_serializer import data_serializer_factory

        original_serializer = data_serializer_factory(
            category="seed-prompt-entries", data_type=self.data_type, value=self.value
        )

        self.value_sha256 = await original_serializer.get_sha256()

    def set_encoding_metadata(self):
        """
        This method sets the encoding data for the prompt within metadata dictionary. For images, this is just the
        file format. For audio and video, this also includes bitrate (kBits/s as int), samplerate (samples/second
        as int), bitdepth (as int), filesize (bytes as int), and duration (seconds as int) if the file type is
        supported by TinyTag. Example suppported file types include: MP3, MP4, M4A, and WAV.
        """
        if self.data_type not in ["audio_path", "video_path", "image_path"]:
            return
        if self.metadata is None:
            self.metadata = {}
        extension = DataTypeSerializer.get_extension(self.value)
        if extension:
            extension = extension.lstrip(".")
            self.metadata.update({"format": extension})
        if self.data_type in ["audio_path", "video_path"]:
            if TinyTag.is_supported(self.value):
                try:
                    tag = TinyTag.get(self.value)
                    self.metadata.update(
                        {
                            "bitrate": int(round(tag.bitrate)),
                            "samplerate": tag.samplerate,
                            "bitdepth": tag.bitdepth,
                            "filesize": tag.filesize,
                            "duration": int(round(tag.duration)),
                        }
                    )
                except Exception as ex:
                    logger.error(f"Error getting audio/video data for {self.value}: {ex}")
            else:
                logger.warning(
                    f"Getting audio/video data via TinyTag is not supported for {self.value}.\
                                If needed, update metadata manually."
                )

    @classmethod
    def from_yaml_with_required_parameters(
        cls, template_path: Union[str, Path], required_parameters: list[str], error_message: Optional[str] = None
    ) -> "SeedPrompt":
        """
        Load a SeedPrompt from a YAML file and validate that it contains specific parameters.

        Args:
            template_path: Path to the YAML file containing the template.
            required_parameters: List of parameter names that must exist in the template.
            error_message: Custom error message if validation fails. If None, a default message is used.

        Returns:
            SeedPrompt: The loaded and validated seed prompt.

        Raises:
            ValueError: If the template doesn't contain all required parameters.
        """
        sp = cls.from_yaml_file(template_path)

        if sp.parameters is None or not all(param in sp.parameters for param in required_parameters):
            if error_message is None:
                error_message = f"Template must have these parameters: {', '.join(required_parameters)}"
            raise ValueError(f"{error_message}: '{sp}'")

        return sp


class SeedPromptGroup(YamlLoadable):
    """
    A group of prompts that need to be sent together, along with an objective.

    This class is useful when a target requires multiple prompts to be grouped
    and sent together. All prompts in the group should share the same `prompt_group_id`. Within the group,
    there can be multiple prompts with the same `prompt_seed_alias`, which will be sent together in a single request.

    """

    prompts: Sequence[SeedPrompt]

    def __init__(
        self,
        *,
        prompts: Union[Sequence[SeedPrompt], Sequence[Dict[str, Any]]],
    ):
        if not prompts:
            raise ValueError("SeedPromptGroup cannot be empty.")
        self.prompts = []
        for prompt in prompts:
            if isinstance(prompt, SeedPrompt):
                self.prompts.append(prompt)
            elif isinstance(prompt, dict):
                self.prompts.append(SeedPrompt(**prompt))

        self._enforce_consistent_group_id()
        self._enforce_consistent_role()

        # check seed_alias and group the seed prompts by sequence if seed_alias is set
        if any(prompt.prompt_seed_alias for prompt in self.prompts):
            self.prompts = sorted(
                self.prompts,
                key=lambda prompt: (prompt.prompt_seed_id, prompt.sequence if prompt.sequence is not None else 0),
            )

    def render_template_value(self, **kwargs):
        """Renders self.value as a template, applying provided parameters in kwargs

        Args:
            kwargs:Key-value pairs to replace in the SeedPromptGroup value.

        Returns:
            None

        Raises:
            ValueError: If parameters are missing or invalid in the template.
        """

        for prompt in self.prompts:
            prompt.value = prompt.render_template_value(**kwargs)

    def _enforce_consistent_group_id(self):
        """
        Ensures that if any of the prompts already have a group ID set,
        they share the same ID. If none have a group ID set, assign a
        new UUID to all prompts.

        Raises:
            ValueError: If multiple different group IDs exist among the prompts.
        """
        existing_group_ids = {prompt.prompt_group_id for prompt in self.prompts if prompt.prompt_group_id is not None}

        if len(existing_group_ids) > 1:
            # More than one distinct group ID found among prompts.
            raise ValueError("Inconsistent group IDs found across prompts.")
        elif len(existing_group_ids) == 1:
            # Exactly one group ID is set; apply it to all.
            group_id = existing_group_ids.pop()
            for prompt in self.prompts:
                prompt.prompt_group_id = group_id
        else:
            # No group IDs set; generate a fresh one and assign it to all.
            new_group_id = uuid.uuid4()
            for prompt in self.prompts:
                prompt.prompt_group_id = new_group_id

    def _enforce_consistent_role(self):
        """
        Ensures that all prompts in the group have the same role.
        If they do not, raises a ValueError.

        Raises:
            ValueError: If multiple different roles exist among the prompts.
        """
        alias_id_to_role = {}
        for prompt in self.prompts:
            role = prompt.role
            alias_id = prompt.prompt_seed_id
            if alias_id in alias_id_to_role:
                if alias_id_to_role[alias_id] != role:
                    raise ValueError(f"Inconsistent roles found across prompts: {role}")
            else:
                alias_id_to_role[alias_id] = role

    def is_single_request(self) -> bool:
        unique_sequences = {prompt.sequence for prompt in self.prompts}
        return len(unique_sequences) <= 1

    def is_single_part_single_text_request(self) -> bool:
        return len(self.prompts) == 1 and self.prompts[0].data_type == "text"

    def __repr__(self):
        return f"<SeedPromptGroup(prompts={len(self.prompts)} prompts)>"


class SeedPromptDataset(YamlLoadable):
    """
    SeedPromptDataset manages seed prompts plus optional top-level defaults.
    Prompts are stored as a Sequence[SeedPrompt], so references to prompt properties
    are straightforward (e.g. ds.prompts[0].value).
    """

    data_type: Optional[str]
    name: Optional[str]
    dataset_name: Optional[str]
    harm_categories: Optional[Sequence[str]]
    description: Optional[str]
    authors: Optional[Sequence[str]]
    groups: Optional[Sequence[str]]
    source: Optional[str]
    date_added: Optional[datetime]
    added_by: Optional[str]

    # Now the actual prompts
    prompts: Sequence["SeedPrompt"]

    def __init__(
        self,
        *,
        prompts: Optional[Union[Sequence[Dict[str, Any]], Sequence[SeedPrompt]]] = None,
        data_type: Optional[PromptDataType] = "text",
        name: Optional[str] = None,
        dataset_name: Optional[str] = None,
        harm_categories: Optional[Sequence[str]] = None,
        description: Optional[str] = None,
        authors: Optional[Sequence[str]] = None,
        groups: Optional[Sequence[str]] = None,
        source: Optional[str] = None,
        date_added: Optional[datetime] = None,
        added_by: Optional[str] = None,
    ):
        """
        Initialize the dataset.
        Typically, you'll call from_dict or from_yaml_file so that top-level defaults
        are merged into each prompt. If you're passing prompts directly, they can be
        either a list of SeedPrompt objects or prompt dictionaries (which then get
        converted to SeedPrompt objects).
        """
        if prompts is None:
            prompts = []
        if not prompts:
            raise ValueError("SeedPromptDataset cannot be empty.")

        # Store top-level fields
        self.data_type = data_type
        self.name = name
        self.dataset_name = dataset_name

        self.harm_categories = harm_categories
        self.description = description
        self.authors = authors or []
        self.groups = groups or []
        self.source = source
        self.date_added = date_added or datetime.now()
        self.added_by = added_by

        # Convert any dictionaries in `prompts` to SeedPrompt objects
        self.prompts = []
        for p in prompts:
            if isinstance(p, dict):
                self.prompts.append(SeedPrompt(**p))
            elif isinstance(p, SeedPrompt):
                self.prompts.append(p)
            else:
                raise ValueError("Prompts should be either dicts or SeedPrompt objects. Got something else.")

    def get_values(self, first: Optional[PositiveInt] = None, last: Optional[PositiveInt] = None) -> Sequence[str]:
        """
        Extracts and returns a list of prompt values from the dataset. By default, returns all of them.

        Args:
            first (Optional[int]): If provided, values from the first N prompts are included.
            last (Optional[int]): If provided, values from the last N prompts are included.

        Returns:
            Sequence[str]: A list of prompt values.
        """
        values = [prompt.value for prompt in self.prompts]

        if first is None and last is None:
            return values
        if first and last and first + last >= len(values):
            return values  # simply return all values in case of an overlap

        first_part = values[:first] if first is not None else []
        last_part = values[-last:] if last is not None else []

        return first_part + last_part

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SeedPromptDataset":
        """
        Builds a SeedPromptDataset by merging top-level defaults into each item in 'prompts'.
        """
        # Pop out the prompts section
        prompts_data = data.pop("prompts", [])
        dataset_defaults = data  # everything else is top-level

        merged_prompts = []
        for p in prompts_data:
            # Merge dataset-level fields with the prompt-level fields
            merged = utils.combine_dict(dataset_defaults, p)

            merged["harm_categories"] = utils.combine_list(
                dataset_defaults.get("harm_categories", []),
                p.get("harm_categories", []),
            )

            merged["authors"] = utils.combine_list(
                dataset_defaults.get("authors", []),
                p.get("authors", []),
            )

            merged["groups"] = utils.combine_list(
                dataset_defaults.get("groups", []),
                p.get("groups", []),
            )

            if "data_type" not in merged:
                merged["data_type"] = dataset_defaults.get("data_type", "text")

            merged_prompts.append(merged)

        for prompt in merged_prompts:
            if "prompt_group_id" in prompt:
                raise ValueError("prompt_group_id should not be set in prompt data")

        SeedPromptDataset._set_prompt_group_id_by_alias(seed_prompts=merged_prompts)
        SeedPromptDataset._set_prompt_seed_id_by_alias(seed_prompts=merged_prompts)

        # Now create the dataset with the newly merged prompt dicts
        return cls(prompts=merged_prompts, **dataset_defaults)

    def render_template_value(self, **kwargs):
        """Renders self.value as a template, applying provided parameters in kwargs

        Args:
            kwargs:Key-value pairs to replace in the SeedPromptDataset value.

        Returns:
            None

        Raises:
            ValueError: If parameters are missing or invalid in the template.
        """

        for prompt in self.prompts:
            prompt.value = prompt.render_template_value(**kwargs)

    @staticmethod
    def _set_id_by_alias(seed_prompts: Sequence[dict], alias_field: str, id_field: str):
        """
        Helper function to set unique IDs for seed prompts based on an alias field.

        """
        alias_to_id = {}

        for prompt in seed_prompts:
            alias = prompt.get(alias_field)
            if alias:
                if alias not in alias_to_id:
                    alias_to_id[alias] = uuid.uuid4()
                prompt[id_field] = alias_to_id[alias]
            else:
                prompt[id_field] = uuid.uuid4()

    @staticmethod
    def _set_prompt_group_id_by_alias(seed_prompts: Sequence[dict]):
        """
        Sets all seed_prompt_group_ids based on prompt_group_id_alias matches
        This is important so the prompt_group_id_alias can be set in yaml to group prompts
        """
        SeedPromptDataset._set_id_by_alias(seed_prompts, alias_field="prompt_group_alias", id_field="prompt_group_id")

    @staticmethod
    def _set_prompt_seed_id_by_alias(seed_prompts: Sequence[dict]):
        """
        Sets all seed_prompt_ids based on prompt_seed_alias matches
        This is important so the prompt_group_id_alias can be set in yaml to group prompts
        """
        SeedPromptDataset._set_id_by_alias(seed_prompts, alias_field="prompt_group_alias", id_field="prompt_group_id")

    @staticmethod
    def group_seed_prompts_by_prompt_group_id(seed_prompts: Sequence[SeedPrompt]) -> Sequence[SeedPromptGroup]:
        """
        Groups the given list of SeedPrompts by their prompt_group_id and creates
        SeedPromptGroup instances.

        Args:
            seed_prompts: A list of SeedPrompt objects.

        Returns:
            A list of SeedPromptGroup objects, with prompts grouped by prompt_group_id.
        """
        # Group seed prompts by `prompt_group_id`
        grouped_prompts = defaultdict(list)
        for prompt in seed_prompts:
            if prompt.prompt_group_id:
                grouped_prompts[prompt.prompt_group_id].append(prompt)
            else:
                grouped_prompts[uuid.uuid4()].append(prompt)

        # Create SeedPromptGroup instances from grouped prompts
        seed_prompt_groups = []
        for group_prompts in grouped_prompts.values():
            if len(group_prompts) > 1:
                group_prompts.sort(key=lambda prompt: prompt.sequence)

            seed_prompt_group = SeedPromptGroup(prompts=group_prompts)
            seed_prompt_groups.append(seed_prompt_group)

        return seed_prompt_groups

    def __repr__(self):
        return f"<SeedPromptDataset(prompts={len(self.prompts)} prompts)>"
