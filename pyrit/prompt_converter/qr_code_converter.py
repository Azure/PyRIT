# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import base64
from io import BytesIO
import qrcode
from typing import Optional

from pyrit.models import PromptDataType
from pyrit.models.data_type_serializer import data_serializer_factory
from pyrit.prompt_converter import PromptConverter, ConverterResult


class QRCodeConverter(PromptConverter):
    """Converts a text string to a QR code image.
    Args:
        version (int, optional): integer from 1 to 40 that controls the size of the QR Code
        (the smallest, version 1, is a 21x21 matrix). Set to None and use the fit parameter
        when making the code to determine size automatically. Defaults to None.
        fit (bool, optional): automatically determine the size of the QR code. Defaults to True.
        box_size (int, optional): controls how many pixels each "box" of the QR code is. Defaults to 10.
        border (optional, int): controls how many boxes thick the border should be. Defaults to 4.
        back_color (optional, tuple): sets background color of QR code, using RGB values.
        Defaults to white: (255, 255, 255).
        fill_color (optional, tuple): sets fill color of QR code, using RGB values. Defaults to black: (0, 0, 0).
        output_filename (optional, str): filename to store QR code. If not provided a unique UUID will be used.
    """

    def __init__(
        self,
        version: Optional[int] = None,
        fit: Optional[bool] = True,
        box_size: Optional[int] = 10,
        border: Optional[int] = 4,
        back_color: Optional[tuple[int, int, int]] = (255, 255, 255),
        fill_color: Optional[tuple[int, int, int]] = (0, 0, 0),
        output_filename: Optional[str] = None,
    ):
        self._version = version
        self._fit = fit
        self._box_size = box_size
        self._border = border
        self._back_color = back_color
        self._fill_color = fill_color
        self._output_filename = output_filename

    async def convert_async(self, *, prompt: str, input_type: PromptDataType = "text") -> ConverterResult:
        """
        Converter that converts strings to QR code images.

        Args:
            prompt (str): The prompt to be converted.
            input_type (PromptDataType): type of data to be converted. Defaults to "text".
        Returns:
            ConverterResult: The filename of the converted QR code image as a ConverterResult Object
        """
        if not self.input_supported(input_type):
            raise ValueError("Input type not supported")
        if not prompt:
            raise ValueError("Please provide valid text value")

        # Instantiate a QRCode object to adjust details
        qr = qrcode.QRCode(version=self._version, box_size=self._box_size, border=self._border)
        # Add the data to the QRCode instance
        qr.add_data(prompt)
        # Compile data into QR code array
        qr.make(fit=self._fit)
        # Convert QR code array to image with specified fill and background colors
        qr_image = qr.make_image(fill_color=self._fill_color, back_color=self._back_color)
        image_bytes = BytesIO()
        qr_image.save(image_bytes, format="png")
        image_str = base64.b64encode(image_bytes.getvalue())
        img_serializer = data_serializer_factory(data_type="image_path")
        img_serializer.save_b64_image(data=image_str, output_filename=self._output_filename)
        return ConverterResult(output_text=img_serializer.value, output_type="image_path")

    def input_supported(self, input_type: PromptDataType) -> bool:
        return input_type == "text"
