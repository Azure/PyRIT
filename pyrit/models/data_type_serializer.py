# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import abc
import base64
import hashlib
import os
import time

from typing import Union, Optional

from pathlib import Path
from mimetypes import guess_type

from azure.storage import BlobServiceClient

from pyrit.common.path import RESULTS_PATH
from pyrit.models import PromptDataType


class StorageIO(abc.ABC):
    @abc.abstractmethod
    def read(self, path: Union[Path, str]) -> bytes: ...

    @abc.abstractmethod
    def write(self, path: Union[Path, str], data: bytes) -> None: ...

    @abc.abstractmethod
    def exists(self, path: Union[Path, str]) -> bool: ...

    @abc.abstractmethod
    def isfile(self, path: Union[Path, str]) -> bool: ...


def data_serializer_factory(
    *,
    data_type: PromptDataType,
    value: Optional[str] = None,
    extension: Optional[str] = None,
    storage_io: Optional[StorageIO] = None,
):
    if storage_io is None:
        storage_io = DiskStorageIO()

    if value:
        if data_type == "text":
            return TextDataTypeSerializer(prompt_text=value, storage_io=storage_io)
        elif data_type == "image_path":
            return ImagePathDataTypeSerializer(prompt_text=value, storage_io=storage_io)
        elif data_type == "audio_path":
            return AudioPathDataTypeSerializer(prompt_text=value, storage_io=storage_io)
        elif data_type == "error":
            return ErrorDataTypeSerializer(prompt_text=value, storage_io=storage_io)
        elif data_type == "url":
            return URLDataTypeSerializer(prompt_text=value, storage_io=storage_io)
        else:
            raise ValueError(f"Data type {data_type} not supported")
    else:
        if data_type == "image_path":
            return ImagePathDataTypeSerializer(extension=extension, storage_io=storage_io)
        elif data_type == "audio_path":
            return AudioPathDataTypeSerializer(extension=extension, storage_io=storage_io)
        else:
            raise ValueError(f"Data type {data_type} without prompt text not supported")


class DiskStorageIO(StorageIO):
    def read(self, path: Union[Path, str]) -> bytes:
        with open(path, "rb") as file:
            return file.read()

    def write(self, path: Union[Path, str], data: bytes) -> None:
        with open(path, "wb") as file:
            file.write(data)

    def exists(self, path: Union[Path, str]) -> bool:
        return os.path.exists(path)

    def isfile(self, path: Union[Path, str]) -> bool:
        return os.path.isfile(path)


class AzureStorageIO(StorageIO):
    def __init__(self, azure_blob_service_client: BlobServiceClient, container_name: Optional[str]) -> None:
        super().__init__()

        self.azure_blob_service_client = azure_blob_service_client
        self.container_name = container_name

    def read(self, path: Path | str) -> bytes:
        blob_client = self.azure_blob_service_client.get_blob_client(container=self.container_name, blob=path)
        return blob_client.download_blob().readall()

    def write(self, path: Union[Path, str], data: bytes) -> None:
        container_client = self.azure_blob_service_client.get_container_client(container=self.container_name)
        container_client.upload_blob(name=path, data=data, overwrite=True)

    def exists(self, path: Path | str) -> bool:
        container_client = self.azure_blob_service_client.get_container_client(container=self.container_name)
        matched_blobs = container_client.list_blobs(name_starts_with=path)
        return len(matched_blobs) > 0 and any(blob.name == str(path) for blob in matched_blobs)

    def isfile(self, path: Path | str) -> bool:
        container_client = self.azure_blob_service_client.get_container_client(container=self.container_name)
        matched_blobs = container_client.list_blobs(name_starts_with=path)
        # TODO: Does this logic really work?
        return len(matched_blobs) > 1 and any(blob.name == str(path) for blob in matched_blobs)


class DataTypeSerializer(abc.ABC):
    """
    Abstract base class for data type normalizers.

    This class is responsible for saving multi-modal types to disk.
    """

    data_type: PromptDataType
    value: str
    data_directory: Path
    file_extension: str
    storage_io: StorageIO

    _file_path: Optional[Path] = None

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
        self.storage_io.write(self.value, data)

    def save_b64_image(self, data: str, output_filename: Optional[str] = None) -> None:
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
        image_bytes = base64.b64decode(data)
        self.storage_io.write(self.value, image_bytes)

    def read_data(self) -> bytes:
        """
        Reads the data from the disk.
        """
        if not self.data_on_disk():
            raise TypeError(f"Data for data Type {self.data_type} is not stored on disk")

        if not self.value:
            raise RuntimeError("Prompt text not set")

        return self.storage_io.read(self.value)

    def read_data_base64(self) -> str:
        """
        Reads the data from the disk.
        """
        byte_array = self.read_data()
        return base64.b64encode(byte_array).decode("utf-8")

    def get_sha256(self) -> str:
        input_bytes: bytes

        if self.data_on_disk():
            input_bytes = self.storage_io.read(self.value)
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

    # TODO: fix where these are called
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
        raise FileNotFoundError(f"No file found at the specified path: {file_path}")

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
    def __init__(self, *, prompt_text: str, storage_io: StorageIO):
        self.data_type = "text"
        self.value = prompt_text
        self.storage_io = storage_io

    def data_on_disk(self) -> bool:
        return False


class ErrorDataTypeSerializer(DataTypeSerializer):
    def __init__(self, *, prompt_text: str, storage_io: StorageIO):
        self.data_type = "error"
        self.value = prompt_text
        self.storage_io = storage_io

    def data_on_disk(self) -> bool:
        return False


class URLDataTypeSerializer(DataTypeSerializer):
    def __init__(self, *, prompt_text: str, storage_io: StorageIO):
        self.data_type = "url"
        self.value = prompt_text
        self.storage_io = storage_io

    def data_on_disk(self) -> bool:
        return False


class ImagePathDataTypeSerializer(DataTypeSerializer):
    def __init__(
        self,
        *,
        storage_io: StorageIO,
        prompt_text: Optional[str] = None,
        extension: Optional[str] = None,
    ):
        self.data_type = "image_path"
        self.data_directory = Path(RESULTS_PATH) / "dbdata" / "images"
        self.file_extension = extension if extension else "png"
        self.storage_io = storage_io

        if prompt_text:
            self.value = prompt_text

            if not self.storage_io.isfile(self.value):
                raise FileNotFoundError(f"File does not exist: {self.value}")

    def data_on_disk(self) -> bool:
        return True


class AudioPathDataTypeSerializer(DataTypeSerializer):
    def __init__(
        self,
        *,
        storage_io: StorageIO,
        prompt_text: Optional[str] = None,
        extension: Optional[str] = None,
    ):
        self.data_type = "audio_path"
        self.data_directory = Path(RESULTS_PATH) / "dbdata" / "audio"
        self.file_extension = extension if extension else "mp3"
        self.storage_io = storage_io

        if prompt_text:
            self.value = prompt_text

            if not self.storage_io.isfile(self.value):
                raise FileNotFoundError(f"File does not exist: {self.value}")

    def data_on_disk(self) -> bool:
        return True
