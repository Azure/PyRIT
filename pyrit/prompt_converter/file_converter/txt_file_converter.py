# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Optional
from pyrit.memory.memory_models import PromptDataType
from pyrit.prompt_converter.file_converter.file_converter import FileConverter
from uuid import uuid4
import json


class TxtFileConverter(FileConverter):
    """Creates a txt file for a given input string.

    Args:
    file_name: file name without file type suffix.
        For example, for "test_file.txt" only "test_file" is needed.
    """

    def __init__(self, file_name: Optional[str] = None):
        self._file_name = file_name

    def convert(self, *, prompt: str, input_type: PromptDataType = "text") -> str:
        """
        Converter that saves the provided string into a txt file.

        Args:
            prompt (str): The prompt to be converted.
        Returns:
            str: The file name.
        """
        if not self.is_supported(input_type):
            raise ValueError("Input type not supported")

        if not self._file_name:
            self._file_name = str(uuid4())
        full_file_name = f"{self._file_name}.txt"
        with open(full_file_name, "w") as file:
            file.write(prompt)
        return json.dumps({"data": prompt, "file_location": full_file_name})

    def is_supported(self, input_type: PromptDataType) -> bool:
        return input_type == "text"
