# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from pyrit.memory.memory_models import PromptDataType
from pyrit.prompt_converter.file_converter.file_converter import FileConverter
from uuid import uuid4
import json


class TxtFileConverter(FileConverter):
    """Creates a txt file for a given input string."""

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

        file_name = f"{str(uuid4())}.txt"
        with open(file_name, "w") as file:
            file.write(prompt)
        return json.dumps({"data": prompt, "file_location": file_name})

    def is_supported(self, input_type: PromptDataType) -> bool:
        return input_type == "text"
