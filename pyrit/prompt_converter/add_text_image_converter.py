# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
import pathlib

from io import BytesIO
from augly.image.transforms import OverlayText
from augly.utils import base_paths

from pyrit.models.data_type_serializer import data_serializer_factory
from pyrit.models.prompt_request_piece import PromptDataType
from pyrit.prompt_converter import PromptConverter, ConverterResult
from PIL import Image

logger = logging.getLogger(__name__)


class AddTextImageConverter(PromptConverter):
    """
    Adds a string to an image

    Args:
        font (optional, str): path of font to use - will set to source sans pro as default
        color (optional, tuple): color to print text in, using RGB values, black is the default if not provided
        font_size (optional, float): size of font to use
        x_pos (optional, float): x coordinate to place text in (0 is left most)
        y_pos (optional, float): y coordinate to place text in (0 is upper most)

    """

    def __init__(
        self,
        font: str = "",
        color: tuple[int, int, int] = (255, 255, 255),
        font_size: float = 0.05,
        x_pos: int = 0,
        y_pos: int = 0,
    ):
        if font:
            self.font = font
        else:
            self.font = pathlib.Path(base_paths.FONTS_DIR) / "SourceSansPro-Black.ttf"
        self.font_size = font_size
        self.color = color
        self.x = x_pos
        self.y = y_pos

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

        text_ascii_rep = []
        # Splits prompt into list[int] representation needed for augly
        text_to_add = kwargs["text_to_add"]
        for word in text_to_add.split("\n"):
            word_list = []
            for letter in word:
                word_list.append(ord(letter) - 32)

            text_ascii_rep.append(word_list)

        output_file = data.get_data_filename()
        try:
            overlay_text = OverlayText(
                text=text_ascii_rep,
                font_file=self.font,
                color=self.color,
                font_size=self.font_size,
                x_pos=self.x,
                y_pos=self.y,
            )

        except OSError:
            logger.error(f"could not load font {self.font}. Switching to default")
            overlay_text = OverlayText(
                text=text_ascii_rep,
                font_file=pathlib.Path(base_paths.FONTS_DIR) / "SourceSansPro-Black.ttf",
                font_size=self.font_size,
                color=self.color,
                x_pos=self.x,
                y_pos=self.y,
            )

        except Exception as e:
            logger.error(f"Error with augly: {e}")
            raise
        new_img = overlay_text.apply_transform(image=original_img)
        new_img.save(fp=output_file)
        return ConverterResult(output_text=output_file, output_type="image_path")

    def input_supported(self, input_type: PromptDataType) -> bool:
        return input_type == "image_path"
