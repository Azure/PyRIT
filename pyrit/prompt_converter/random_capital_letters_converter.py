# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
import random

from pyrit.models import PromptDataType
from pyrit.prompt_converter import ConverterResult, PromptConverter

logger = logging.getLogger(__name__)


class RandomCapitalLettersConverter(PromptConverter):
    """This converter takes a prompt and randomly capitalizes it by a percentage of the total characters.

    Args:
        This accepts a text prompt, and a percentage of randomization from 1 to 100.  This includes decimal
        points in that range.
    """

    def __init__(self, percentage: float = 100.0) -> None:
        self.percentage = percentage

    def input_supported(self, input_type: PromptDataType) -> bool:
        return input_type == "text"

    def output_supported(self, output_type: PromptDataType) -> bool:
        return output_type == "text"

    # function to check if character is lower case returns True or False
    def is_lowercase_letter(self, char):
        return char.islower()

    # function to check if number is between 1 and 100 returns True or False
    def is_percentage(self, input_string):
        try:
            number = float(input_string)
            return 1 <= number <= 100
        except ValueError:
            return False

    # function to generate an array of random positions set by a number
    def generate_random_positions(self, total_length, set_number):
        # Ensure the set number is not greater than the total length
        if set_number > total_length:
            logger.error(f"Set number {set_number} cannot be greater than the total length which is {total_length}.")
            raise ValueError(
                f"Set number {set_number} cannot be greater than the total length which is {total_length}."
            )

        # Generate a list of unique random positions
        random_positions = random.sample(range(total_length), set_number)

        return random_positions

    def string_to_upper_case_by_percentage(self, percentage, prompt):
        if not self.is_percentage(percentage):
            logger.error(f"Percentage number {percentage} cannot be higher than 100 and lower than 1.")
            raise ValueError(f"Percentage number {percentage} cannot be higher than 100 and lower than 1.")
        target_count = int(len(prompt) * (percentage / 100))
        random_positions = self.generate_random_positions(len(prompt), target_count)
        output = list(prompt)
        for pos in random_positions:
            if self.is_lowercase_letter(prompt[pos]):
                output[pos] = prompt[pos].upper()
        return "".join(output)

    async def convert_async(self, *, prompt: str, input_type: PromptDataType = "text") -> ConverterResult:
        """
        Simple converter that converts the prompt to capital letters via a percentage .
        """
        if not self.input_supported(input_type):
            raise ValueError("Input type not supported")

        output = self.string_to_upper_case_by_percentage(self.percentage, prompt)
        return ConverterResult(output_text=output, output_type="text")
