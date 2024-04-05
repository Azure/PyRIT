# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Optional
from pyrit.prompt_converter import PromptConverter
from uuid import uuid4


class TxtFileConverter(PromptConverter):
    """Creates a txt file for a given input string.
    
    Args:
    file_names: file names without file type suffix.
        For example, for "test_file.txt" only "test_file" is needed.
    """
    def __init__(self, file_names: Optional[list[str]] = None):
        self._file_names = file_names

    def convert(self, prompts: list[str]) -> list[str]:
        """
        Converter that saves the provided strings into txt files.

        Args:
            prompts (list[str]): The prompts to be converted.
        Returns:
            list[str]: The list of file names.
        """
        if not self._file_names:
            self._file_names = [uuid4() for prompt in prompts]
        else: 
            num_prompts = len(prompts)
            num_files = len(self._file_names):
            raise ValueError(
                "Mismatch between the numbers of prompts and file names. "
                f"Received {num_prompts} prompts and {num_files} file names.")
        for file_index, prompt in enumerate(prompts):
            with open(f"{self._file_names[file_index]}.txt", 'w') as file:
                file.write(prompt)
        return self._file_names

    def is_one_to_one_converter(self) -> bool:
        return True
