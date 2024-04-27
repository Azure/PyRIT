# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from pyrit.models.prompt_request_piece import PromptDataType
from pyrit.prompt_converter import PromptConverter, ConverterResult
from PIL import Image, ImageDraw, ImageFont


class AddTextImageConverter(PromptConverter):
    """Adds a string to an image"""

    def __init__(self, input_file: str, output_file: str = "edited_image.png"):
        self.input_file = input_file
        self.output_file = output_file

    def convert(self, *, prompt: str, input_type: PromptDataType = "image_path") -> ConverterResult:
        """
        Converter that uses art to convert strings to ASCII art.
        This can sometimes bypass LLM filters

        Args:
            prompt (str): The prompt to be converted.
        Returns:
            str: The converted prompt.
        """
        if not self.input_supported(input_type):
            raise ValueError("Input type not supported")

        # Open the image
        original_img = Image.open(self.input_file)

        # Specify the font and text for the bottom text
        font = ImageFont.load_default()

        # Calculate the width of the text
        text_width = font.getlength(prompt)

        # Calculate the dimensions for the new image with space for text
        text_height = 100
        new_width = original_img.width
        new_height = original_img.height + text_height

        # Create a new image with white background and old photo
        new_img = Image.new("RGB", (new_width, new_height), (255, 255, 255))
        new_img.paste(original_img, (0, 0))
        draw = ImageDraw.Draw(new_img)

        # Calculate text position at the bottom center
        text_x = (new_width - text_width) // 2
        text_y = original_img.height + 10  # 10 pixels below the original image

        # Draw text and save
        draw.text((text_x, text_y), prompt, fill=(0, 0, 0), font=font)
        new_img.save(self.output_file)
        # return str(self.output_file)  # returns output file name for now
        return ConverterResult(output_text=str(self.output_file), output_type="image_path")

    def input_supported(self, input_type: PromptDataType) -> bool:
        return input_type == "image_path"
