# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

import logging
import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Sequence, Union

from jinja2 import BaseLoader, Environment, StrictUndefined, Template, Undefined
from tinytag import TinyTag

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
    prompt_group_id (Optional[uuid.UUID]): a unique identifier generated for the prompt group this prompt belongs to.
    prompt_group_alias (Optional[str]): user set alias for the prompt group. Prompts in the same group will be sent
    together. If the group alias is not set, the prompt is assumed to be the only prompt in the group.
    sequence (Optional[int]): Sequence number for ordering prompts with the same seed alias. Relevant only if the prompt
    seed alias is set and there is more than one prompt associated with that alias. Defaults to 0.
    role: (ChatMessageRole): The role of the prompt in a chat context--"system", "user", or "assistant". Defaults to
    "user".
    prompt_seed_alias (Optional[str]): An alias for the seed prompt, used to identify which seed prompts should be
    sent in the same turn. This is useful for multimodal prompts where multiple prompts need to be sent
    together in a single turn. If not set, the prompt is assumed to be the only prompt in the turn.
    prompt_seed_alias_id (Optional[uuid.UUID]): A unique identifier for the prompt seed alias.

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
