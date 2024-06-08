# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import abc
import base64
import hashlib
import os
import time

from pathlib import Path
from mimetypes import guess_type

from pyrit.common.path import RESULTS_PATH
from pyrit.models import PromptDataType


def data_serializer_factory(*, data_type: PromptDataType, value: str = None, extension: str = None):
    if value:
        if data_type == "text":
            return TextDataTypeSerializer(prompt_text=value)
        elif data_type == "image_path":
            return ImagePathDataTypeSerializer(prompt_text=value)
        elif data_type == "audio_path":
            return AudioPathDataTypeSerializer(prompt_text=value)
        elif data_type == "error":
            return ErrorDataTypeSerializer(prompt_text=value)
        elif data_type == "url":
            return URLDataTypeSerializer(prompt_text=value)
        else:
            raise ValueError(f"Data type {data_type} not supported")
    else:
        if data_type == "image_path":
            return ImagePathDataTypeSerializer(extension=extension)
        elif data_type == "audio_path":
            return AudioPathDataTypeSerializer(extension=extension)
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

    _file_path: Path = None

    @abc.abstractmethod
    def data_on_disk(self) -> bool:
        """
        Returns True if the data is stored on disk.
        """
        pass

    def save_data(self, data: bytes) -> None:
        """
        Saves the data to disk.
        """
        self.value = str(self.get_data_filename())
        with open(self.value, "wb") as file:
            file.write(data)

    def save_b64_image(self, data: str, output_filename: str = None) -> None:
        """
        Saves the base64 encoded image to disk.
        Arguments:
            data: string with base64 data
            output_filename (optional, str): filename to store image as. Defaults to UUID if not provided
        """
        if output_filename:
            self.value = output_filename
        else:
            self.value = str(self.get_data_filename())
        with open(self.value, "wb") as file:
            image_bytes = base64.b64decode(data)
            file.write(image_bytes)

    def read_data(self) -> bytes:
        """
        Reads the data from the disk.
        """
        if not self.data_on_disk():
            raise TypeError(f"Data for data Type {self.data_type} is not stored on disk")

        if not self.value:
            raise RuntimeError("Prompt text not set")

        with open(self.value, "rb") as file:
            return file.read()

    def read_data_base64(self) -> str:
        """
        Reads the data from the disk.
        """
        byte_array = self.read_data()
        return base64.b64encode(byte_array).decode("utf-8")

    def get_sha256(self) -> str:
        input_bytes: bytes

        if self.data_on_disk():
            with open(self.value, "rb") as file:
                input_bytes = file.read()
        else:
            input_bytes = self.value.encode("utf-8")

        hash_object = hashlib.sha256(input_bytes)
        return hash_object.hexdigest()

    def get_data_filename(self) -> Path:
        """
        Generates or retrieves a unique filename for the data file.
        """
        if self._file_path:
            return self._file_path

        if not self.data_on_disk():
            raise TypeError("Data is not stored on disk")

        if not self.data_directory:
            raise RuntimeError("Data directory not set")

        if not self.data_directory.exists():
            self.data_directory.mkdir(parents=True, exist_ok=True)

        ticks = int(time.time() * 1_000_000)
        self._file_path = Path(self.data_directory, f"{ticks}.{self.file_extension}")
        return self._file_path

    @staticmethod
    def path_exists(file_path: str) -> bool:
        """
        Check if the file exists at the specified path.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"No file found at the specified path: {file_path}")
        return True

    @staticmethod
    def get_extension(file_path: str) -> str | None:
        """
        Get the file extension from the file path.
        """
        if DataTypeSerializer.path_exists(file_path):
            _, ext = os.path.splitext(file_path)
            return ext
        return None

    @staticmethod
    def get_mime_type(file_path: str) -> str | None:
        """
        Get the MIME type of the file path.
        """
        if DataTypeSerializer.path_exists(file_path):
            mime_type, _ = guess_type(file_path)
            return mime_type
        return None


class TextDataTypeSerializer(DataTypeSerializer):
    def __init__(self, *, prompt_text: str):
        self.data_type = "text"
        self.value = prompt_text

    def data_on_disk(self) -> bool:
        return False


class ErrorDataTypeSerializer(DataTypeSerializer):
    def __init__(self, *, prompt_text: str):
        self.data_type = "error"
        self.value = prompt_text

    def data_on_disk(self) -> bool:
        return False


class URLDataTypeSerializer(DataTypeSerializer):
    def __init__(self, *, prompt_text: str):
        self.data_type = "url"
        self.value = prompt_text

    def data_on_disk(self) -> bool:
        return False


class ImagePathDataTypeSerializer(DataTypeSerializer):
    def __init__(self, *, prompt_text: str = None, extension: str = None):
        self.data_type = "image_path"
        self.data_directory = Path(RESULTS_PATH) / "dbdata" / "images"
        self.file_extension = extension if extension else "png"

        if prompt_text:
            self.value = prompt_text

            if not os.path.isfile(self.value):
                raise FileNotFoundError(f"File does not exist: {self.value}")

    def data_on_disk(self) -> bool:
        return True


class AudioPathDataTypeSerializer(DataTypeSerializer):
    def __init__(self, *, prompt_text: str = None, extension: str = None):
        self.data_type = "audio_path"
        self.data_directory = Path(RESULTS_PATH) / "dbdata" / "audio"
        self.file_extension = extension if extension else "mp3"

        if prompt_text:
            self.value = prompt_text

            if not os.path.isfile(self.value):
                raise FileNotFoundError(f"File does not exist: {self.value}")

    def data_on_disk(self) -> bool:
        return True
