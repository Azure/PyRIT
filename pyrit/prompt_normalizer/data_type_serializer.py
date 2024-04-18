# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import abc
import base64
import os
import time

from pathlib import Path

from pyrit.common.path import RESULTS_PATH
from pyrit.models import PromptDataType


def data_serializer_factory(*, data_type: PromptDataType, prompt_text: str = None):
    if prompt_text:
        if data_type == "text":
            return TextDataTypeSerializer(prompt_text=prompt_text)
        elif data_type == "image_path":
            return ImagePathDataTypeSerializer(prompt_text=prompt_text)
        else:
            raise ValueError(f"Data type {data_type} not supported")
    else:
        if data_type == "image_path":
            return ImagePathDataTypeSerializer()
        else:
            raise ValueError(f"Data type {data_type} without prompt text not supported")


class DataTypeSerializer(abc.ABC):
    """
    Abstract base class for data type normalizers.

    This class is responsible for saving multi-modal types to disk.
    """

    data_type: PromptDataType
    prompt_text: str
    data_directory: Path
    file_extension: str

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
        self.prompt_text = str(self._get_data_filename())
        with open(self.prompt_text, "wb") as file:
            file.write(data)

    def read_data(self) -> bytes:
        """
        Reads the data from the disk.
        """
        if not self.data_on_disk():
            raise TypeError(f"Data for data Type {self.data_type} is not stored on disk")

        if not self.prompt_text:
            raise RuntimeError("Prompt text not set")

        with open(self.prompt_text, "rb") as file:
            return file.read()

    def read_data_base64(self) -> str:
        """
        Reads the data from the disk.
        """
        byte_array = self.read_data()
        return base64.b64encode(byte_array).decode("utf-8")

    def _get_data_filename(self) -> Path:
        """
        Generates a unique filename for the data file.
        """
        if not self.data_on_disk():
            raise TypeError("Data is not stored on disk")

        if not self.data_directory:
            raise RuntimeError("Data directory not set")

        if not self.data_directory.exists():
            self.data_directory.mkdir(parents=True, exist_ok=True)

        ticks = int(time.time() * 1_000_000)
        return Path(self.data_directory, f"{ticks}{self.file_extension}")


class TextDataTypeSerializer(DataTypeSerializer):
    def __init__(self, *, prompt_text: str):
        self.data_type = "text"
        self.prompt_text = prompt_text

    def data_on_disk(self) -> bool:
        return False


class ImagePathDataTypeSerializer(DataTypeSerializer):
    def __init__(self, *, prompt_text: str = None, extension: str = ".png"):
        self.data_type = "image_path"
        self.data_directory = Path(RESULTS_PATH) / "dbdata" / "images"
        self.file_extension = extension

        if prompt_text:
            self.prompt_text = prompt_text

            if not os.path.isfile(self.prompt_text):
                raise FileNotFoundError(f"File does not exist: {self.prompt_text}")

    def data_on_disk(self) -> bool:
        return True
