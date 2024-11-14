# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
import base64
import string
from typing import Optional

from PIL import Image, ImageDraw, ImageFont
import textwrap
from io import BytesIO

from pyrit.models import data_serializer_factory
from pyrit.models import PromptDataType
from pyrit.prompt_converter import PromptConverter, ConverterResult

logger = logging.getLogger(__name__)


class AddImageTextConverter(PromptConverter):
    """
    Adds a string to an image and wraps the text into multiple lines if necessary.
    This class is similar to AddImageTextConverter except
    we pass in an image file path as an argument to the constructor as opposed to text.

    Args:
        img_to_add (str): File path of image to add text to
        font_name (str, Optional): Path of font to use. Must be a TrueType font (.ttf). Defaults to "arial.ttf".
        color (tuple, Optional): Color to print text in, using RGB values. Defaults to (0, 0, 0).
        font_size (float, Optional): Size of font to use. Defaults to 15.
        x_pos (int, Optional): X coordinate to place text in (0 is left most). Defaults to 10.
        y_pos (int, Optional): Y coordinate to place text in (0 is upper most). Defaults to 10.
    """

    def __init__(
        self,
        img_to_add: str,
        font_name: Optional[str] = "arial.ttf",
        color: Optional[tuple[int, int, int]] = (0, 0, 0),
        font_size: Optional[int] = 15,
        x_pos: Optional[int] = 10,
        y_pos: Optional[int] = 10,
    ):
        if not img_to_add:
            raise ValueError("Please provide valid image path")
        if not font_name.endswith(".ttf"):
            raise ValueError("The specified font must be a TrueType font with a .ttf extension")
        self._img_to_add = img_to_add
        self._font_name = font_name
        self._font_size = font_size
        self._font = self._load_font()
        self._color = color
        self._x_pos = x_pos
        self._y_pos = y_pos

    def _load_font(self):
        """
        Load the font for a given font name and font size

        Returns:
        ImageFont.FreeTypeFont or ImageFont.ImageFont: The loaded font object. If the specified font
        cannot be loaded, the default font is returned.

        Raises:
            OSError: If the font resource cannot be loaded, a warning is logged and the default font is used instead.
        """
        # Try to load the specified font
        try:
            font = ImageFont.truetype(self._font_name, self._font_size)
        except OSError:
            logger.warning(f"Cannot open font resource: {self._font_name}. Using default font.")
            font = ImageFont.load_default()
        return font

    def _add_text_to_image(self, text: str) -> Image.Image:
        """
        Adds wrapped text to the image at self._img_to_add.

        Args:
            text (str): The text to add to the image.

        Returns:
            Image.Image: The image with added text.
        """
        if not text:
            raise ValueError("Please provide valid text value")
        # Open the image and create a drawing object
        image = Image.open(self._img_to_add)
        draw = ImageDraw.Draw(image)

        # Calculate the maximum width in pixels with margin into account
        margin = 5
        max_width_pixels = image.size[0] - margin

        # Estimate the maximum chars that can fit on a line
        alphabet_letters = string.ascii_letters  # This gives 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
        bbox = draw.textbbox((0, 0), alphabet_letters, font=self._font)
        avg_char_width = (bbox[2] - bbox[0]) / len(alphabet_letters)
        max_chars_per_line = int(max_width_pixels // avg_char_width)

        # Wrap the text
        wrapped_text = textwrap.fill(text, width=max_chars_per_line)

        # Add wrapped text to image
        y_offset = float(self._y_pos)
        for line in wrapped_text.split("\n"):
            draw.text((self._x_pos, y_offset), line, font=self._font, fill=self._color)
            bbox = draw.textbbox((self._x_pos, y_offset), line, font=self._font)
            line_height = bbox[3] - bbox[1]
            y_offset += line_height

        return image

    async def convert_async(self, *, prompt: str, input_type: PromptDataType = "text") -> ConverterResult:
        """
        Converter that overlays input text on the img_to_add.

        Args:
            prompt (str): The prompt to be added to the image.
            input_type (PromptDataType): type of data
        Returns:
            ConverterResult: The filename of the converted image as a ConverterResult Object
        """
        if not self.input_supported(input_type):
            raise ValueError("Input type not supported")

        img_serializer = data_serializer_factory(value=self._img_to_add, data_type="image_path")

        # Add text to the image
        updated_img = self._add_text_to_image(text=prompt)

        image_bytes = BytesIO()
        mime_type = img_serializer.get_mime_type(self._img_to_add)
        image_type = mime_type.split("/")[-1]
        updated_img.save(image_bytes, format=image_type)
        image_str = base64.b64encode(image_bytes.getvalue())
        # Save image as generated UUID filename
        await img_serializer.save_b64_image(data=image_str)
        return ConverterResult(output_text=str(img_serializer.value), output_type="image_path")

    def input_supported(self, input_type: PromptDataType) -> bool:
        return input_type == "text"
