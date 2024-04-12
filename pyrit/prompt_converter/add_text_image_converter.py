# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from pyrit.memory.memory_models import PromptDataType
from pyrit.prompt_converter import PromptConverter
from PIL import Image, ImageDraw, ImageFont


class AddTextImageConverter(PromptConverter):
    """Adds a string to an image"""

    def __init__(self, input_file: str, output_file: str = "edited_image.png"):
        self.input_file = input_file
        self.output_file = output_file

    def convert(self, *, prompt: str, input_type: PromptDataType = "image_file_name") -> str:
        """
        Converter that uses art to convert strings to ASCII art.
        This can sometimes bypass LLM filters

        Args:
            prompt (str): The prompt to be converted.
        Returns:
            str: The converted prompt.
        """
        if not self.is_supported(input_type):
            raise ValueError("Input type not supported")

        original_img = Image.open(self.input_file)

        # Calculate the dimensions for the new image with space for text
        text_height = 100
        new_width = original_img.width
        new_height = original_img.height + text_height

        # Create a new image for the modified version
        new_img = Image.new("RGB", (new_width, new_height), (255, 255, 255))
        new_img.paste(original_img, (0, 0))
        draw = ImageDraw.Draw(new_img)

        # Specify the font, text size, and position at bottom center
        font = ImageFont.load_default()
        text_width, text_height = draw.textsize(prompt, font=font)
        text_x = (new_width - text_width) // 2
        text_y = original_img.height + 10

        draw.text((text_x, text_y), prompt, fill=(0, 0, 0), font=font)

        # Save the new image
        new_img.save(self.output_file)
        return self.output_file  # returns output file name for now

    def is_supported(self, input_type: PromptDataType) -> bool:
        return input_type == "image_file_name"
