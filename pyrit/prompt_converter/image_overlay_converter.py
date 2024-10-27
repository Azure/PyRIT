# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import base64
from typing import Optional

from PIL import Image
from io import BytesIO

from pyrit.models import data_serializer_factory
from pyrit.models import PromptDataType
from pyrit.prompt_converter import PromptConverter, ConverterResult
from pyrit.memory import MemoryInterface, DuckDBMemory


class ImageOverlayConverter(PromptConverter):
    """
    A converter that takes in a base image, and a secondary image to embed within the main image.

    Args:
        base_image_path (str): File path of the base image
        x_pos (int, optional): X coordinate to place second image on the base image (0 is left most). Defaults to 0.
        y_pos (int, optional): Y coordinate to place second image on the base image (0 is upper most). Defaults to 0.
        memory: (memory, optional): Memory to store the chat messages. DuckDBMemory will be used by default.
    """

    def __init__(
            self,
            base_image_path: str,
            x_pos: Optional[int] = 0,
            y_pos: Optional[int] = 0,
            memory: Optional[MemoryInterface] = None,
    ):
        if not base_image_path:
            raise ValueError("Please provide valid image path")

        self._base_image_path = base_image_path
        self._x_pos = x_pos
        self._y_pos = y_pos
        self._memory = memory or DuckDBMemory()

    def _add_overlay_image(self, overlay_image: Image.Image) -> Image.Image:
        """
        Embed the second image onto the base image

        Args:
            overlay_image(Image.Image): The second image to lay on the base one.

        Returns:
            Image.Image: The combined image with overlay.
        """
        if not overlay_image:
            raise ValueError("Please provide a valid image")
        # Open the images
        base_image = Image.open(self._base_image_path)

        # Make a copy of base image, in case the user want to keep their input images unchanged
        copied_base_image = base_image.copy()

        # Paste the second image onto the base image
        copied_base_image.paste(overlay_image, (self._x_pos, self._y_pos), overlay_image)

        return copied_base_image

    async def convert_async(self, *, prompt: str, input_type: PromptDataType = "image_path") -> ConverterResult:
        """
        Converter the base image to embed the second image onto it.

        Args:
            prompt (str): The filename of the second image
            input_type (PromptDataType): type of data, should be image_path

        Returns:
            ConverterResult: converted image with file path
        """
        if not self.input_supported(input_type):
            raise ValueError("Input type not supported")

        img_serializer = data_serializer_factory(value=prompt, data_type="image_path", memory=self._memory)
        second_img_bytes = await img_serializer.read_data()
        second_img = Image.open(BytesIO(second_img_bytes))

        # Add overlay to the base image
        updated_img = self._add_overlay_image(second_img)

        # Encode the image using base64 and return ConverterResult
        # I took the following code from add_image_text_converter.py @author rdheekonda
        image_bytes = BytesIO()
        mime_type = img_serializer.get_mime_type(prompt)
        image_type = mime_type.split("/")[-1]
        updated_img.save(image_bytes, format=image_type)
        image_str = base64.b64encode(image_bytes.getvalue())
        # Save image as generated UUID filename
        await img_serializer.save_b64_image(data=image_str)
        return ConverterResult(output_text=str(img_serializer.value), output_type="image_path")

    def input_supported(self, input_type: PromptDataType) -> bool:
        return input_type == "image_path"
