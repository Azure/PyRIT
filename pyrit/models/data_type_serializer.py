# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

import abc
import base64
import hashlib
import os
import time
from typing import Optional, TYPE_CHECKING, Union
from pathlib import Path
from mimetypes import guess_type
from urllib.parse import urlparse

from pyrit.models.literals import PromptDataType

if TYPE_CHECKING:
    from pyrit.memory import MemoryInterface


def data_serializer_factory(
    *,
    data_type: PromptDataType,
    value: Optional[str] = None,
    extension: Optional[str] = None,
    memory: MemoryInterface = None,
):
    if not memory:
        raise ValueError("The memory provided is invalid. Please provide a valid memory value.")
    if value:
        if data_type == "text":
            return TextDataTypeSerializer(prompt_text=value, memory=memory)
        elif data_type == "image_path":
            return ImagePathDataTypeSerializer(prompt_text=value, memory=memory)
        elif data_type == "audio_path":
            return AudioPathDataTypeSerializer(prompt_text=value, memory=memory)
        elif data_type == "error":
            return ErrorDataTypeSerializer(prompt_text=value, memory=memory)
        elif data_type == "url":
            return URLDataTypeSerializer(prompt_text=value, memory=memory)
        else:
            raise ValueError(f"Data type {data_type} not supported")
    else:
        if data_type == "image_path":
            return ImagePathDataTypeSerializer(extension=extension, memory=memory)
        elif data_type == "audio_path":
            return AudioPathDataTypeSerializer(extension=extension, memory=memory)
        elif data_type == "error":
            return ErrorDataTypeSerializer(prompt_text="", memory=memory)
        else:
            raise ValueError(f"Data type {data_type} without prompt text not supported")


class DataTypeSerializer(abc.ABC):
    """
    Abstract base class for data type normalizers.

    This class is responsible for saving multi-modal types to disk.
    """

    data_type: PromptDataType
    value: str
    data_directory: Path
    file_extension: str
    _memory: MemoryInterface = None

    _file_path: Path = None

    @abc.abstractmethod
    def data_on_disk(self) -> bool:
        """
        Returns True if the data is stored on disk.
        """
        pass

    async def save_data(self, data: bytes) -> None:
        """
        Saves the data to storage.
        """
        self.value = str(await self.get_data_filename())
        await self._memory._storage_io.write_file(self.value, data)

    async def save_b64_image(self, data: str, output_filename: str = None) -> None:
        """
        Saves the base64 encoded image to storage.
        Arguments:
            data: string with base64 data
            output_filename (optional, str): filename to store image as. Defaults to UUID if not provided
        """
        if output_filename:
            self.value = output_filename
        else:
            self.value = str(await self.get_data_filename())
        image_bytes = base64.b64decode(data)
        await self._memory._storage_io.write_file(self.value, image_bytes)

    async def read_data(self) -> bytes:
        """
        Reads the data from the storage.
        """
        if not self.data_on_disk():
            raise TypeError(f"Data for data Type {self.data_type} is not stored on disk")

        if not self.value:
            raise RuntimeError("Prompt text not set")
        # Check if path exists
        file_exists = await self._memory._storage_io.path_exists(path=self.value)
        if not file_exists:
            raise FileNotFoundError(f"File not found: {self.value}")
        # Read the contents from the path
        return await self._memory._storage_io.read_file(self.value)

    async def read_data_base64(self) -> str:
        """
        Reads the data from the storage.
        """
        byte_array = await self.read_data()
        return base64.b64encode(byte_array).decode("utf-8")

    async def get_sha256(self) -> str:
        input_bytes: bytes = None

        if self.data_on_disk():
            input_bytes = await self._memory._storage_io.read_file(self.value)
        else:
            input_bytes = self.value.encode("utf-8")

        hash_object = hashlib.sha256(input_bytes)
        return hash_object.hexdigest()

    async def get_data_filename(self) -> Union[Path, str]:
        """
        Generates or retrieves a unique filename for the data file.
        """
        if self._file_path:
            return self._file_path

        if not self.data_on_disk():
            raise TypeError("Data is not stored on disk")

        if not self.data_directory:
            raise RuntimeError("Data directory not set")

        await self._memory._storage_io.create_directory_if_not_exists(self.data_directory)

        ticks = int(time.time() * 1_000_000)
        if self.is_url(str(self.data_directory)):
            self._file_path = Path(str(self.data_directory) + f"/{ticks}.{self.file_extension}")
        else:
            self._file_path = Path(self.data_directory, f"{ticks}.{self.file_extension}")
        return self._file_path

    @staticmethod
    def get_extension(file_path: str) -> str | None:
        """
        Get the file extension from the file path.
        """
        _, ext = os.path.splitext(file_path)
        return ext if ext else None

    @staticmethod
    def get_mime_type(file_path: str) -> str | None:
        """
        Get the MIME type of the file path.
        """
        mime_type, _ = guess_type(file_path)
        return mime_type

    def is_url(self, path: str) -> bool:
        """
        Helper function to check if a given path is a URL.
        """
        return urlparse(path).scheme in ("http", "https")


class TextDataTypeSerializer(DataTypeSerializer):
    def __init__(self, *, prompt_text: str, memory: MemoryInterface):
        self.data_type = "text"
        self.value = prompt_text
        self._memory = memory

    def data_on_disk(self) -> bool:
        return False


class ErrorDataTypeSerializer(DataTypeSerializer):
    def __init__(self, *, prompt_text: str, memory: MemoryInterface):
        self.data_type = "error"
        self.value = prompt_text
        self._memory = memory

    def data_on_disk(self) -> bool:
        return False


class URLDataTypeSerializer(DataTypeSerializer):
    def __init__(self, *, prompt_text: str, memory: MemoryInterface):
        self.data_type = "url"
        self.value = prompt_text
        self._memory = memory

    def data_on_disk(self) -> bool:
        return False


class ImagePathDataTypeSerializer(DataTypeSerializer):
    def __init__(
        self, *, memory: MemoryInterface = None, prompt_text: Optional[str] = None, extension: Optional[str] = None
    ):
        self._memory = memory
        self.data_type = "image_path"
        self.data_directory = Path(self._memory.results_path + "/dbdata/images")
        self.file_extension = extension if extension else "png"

        if prompt_text:
            self.value = prompt_text

    def data_on_disk(self) -> bool:
        return True


class AudioPathDataTypeSerializer(DataTypeSerializer):
    def __init__(
        self,
        *,
        memory: MemoryInterface = None,
        prompt_text: Optional[str] = None,
        extension: Optional[str] = None,
    ):
        self.data_type = "audio_path"
        self._memory = memory
        self.data_directory = Path(self._memory.results_path + "/dbdata/audio")
        self.file_extension = extension if extension else "mp3"

        if prompt_text:
            self.value = prompt_text

    def data_on_disk(self) -> bool:
        return True
