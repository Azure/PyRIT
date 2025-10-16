# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

import logging
import os
import uuid
from dataclasses import dataclass
from typing import Optional

from tinytag import TinyTag

from pyrit.common.path import PATHS_DICT
from pyrit.models import DataTypeSerializer
from pyrit.models.literals import ChatMessageRole
from pyrit.models.seed import Seed

logger = logging.getLogger(__name__)


@dataclass
class SeedPrompt(Seed):
    """Represents a seed prompt with various attributes and metadata."""

    # Unique identifier for the prompt group
    prompt_group_id: Optional[uuid.UUID] = None

    # Alias for the prompt group
    prompt_group_alias: Optional[str] = None

    # Role of the prompt in a conversation (e.g., "user", "assistant")
    role: Optional[ChatMessageRole] = None

    # Sequence number for ordering prompts in a conversation, prompts with
    # the same sequence number are grouped together if they also share the same prompt_group_id
    sequence: Optional[int] = 0

    # Whether this prompt should be used as the objective for an attack
    use_as_objective: Optional[bool] = None

    def __post_init__(self) -> None:
        """Post-initialization to render the template to replace existing values"""
        self.value = self.render_template_value_silent(**PATHS_DICT)

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
