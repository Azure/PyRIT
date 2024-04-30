# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import base64
from io import BytesIO
import logging

from pyrit.models.data_type_serializer import data_serializer_factory
from pyrit.models.prompt_request_piece import PromptDataType
from pyrit.prompt_converter import PromptConverter, ConverterResult
from PIL import Image, ImageDraw, ImageFont

logger = logging.getLogger(__name__)


class AddTextImageConverter(PromptConverter):
    """
    Adds a string to an image

    Args:
        font (optional, str): font to use - will load default font if not provided
        font_size (optional, float): size of font to use
        color (optional, tuple): color to print text in, using RGB values, black is the default if not provided
    """

    def __init__(self, font: str = "", font_size: float = 0.1, color: tuple[int, int, int] = (255, 255, 255)):
        if font:
            try:
                self.font = ImageFont.load(font)
            except Exception:
                self.font = ImageFont.load_default()
        else:
            self.font = ImageFont.load_default()
        self.font_size = font_size
        self.color = color

    def convert(self, *, prompt: str, input_type: PromptDataType = "image_path", **kwargs) -> ConverterResult:
        """
        Converter that adds text to an image

        Args:
            prompt (str): The prompt to be added to the image.
            input_type (PromptDataType): type of data
            kwargs: dictionary with input "text_to_add"
        Returns:
            ConverterResult: The filename of the converted image as a ConverterResult Object
        """
        if not self.input_supported(input_type):
            raise ValueError("Input type not supported")

        data = data_serializer_factory(value=prompt, data_type="image_path")
        original_img_bytes = data.read_data()
        # Open the image
        original_img = Image.open(BytesIO(original_img_bytes))

        # Calculate the width of the text
        try:
            text_to_add = kwargs["text_to_add"]
        except KeyError:
            logger.info("No text was provided to addTextImageConverter, returning original image")
            return ConverterResult(output_text=prompt, output_type="image_path")

        text_width = self.font.getlength(text_to_add)

        # Calculate the dimensions for the new image with space for text
        text_height = 100
        new_width = original_img.width
        new_height = original_img.height + text_height

        # Create a new image with white background and old photo
        new_img = Image.new("RGB", (new_width, new_height), self.color)
        new_img.paste(original_img, (0, 0))
        draw = ImageDraw.Draw(new_img)

        # Calculate text position at the bottom center
        text_x = (new_width - text_width) // 2
        text_y = original_img.height + 10  # 10 pixels below the original image

        # Draw text and save
        draw.text((text_x, text_y), text_to_add, fill=(0, 0, 0), font=self.font)

        image_bytes = BytesIO()
        new_img.save(image_bytes, format="png")
        image_str = base64.b64encode(image_bytes.getvalue())

        data.save_b64_image(data=image_str)

        return ConverterResult(output_text=str(data.value), output_type="image_path")

    def input_supported(self, input_type: PromptDataType) -> bool:
        return input_type == "image_path"
