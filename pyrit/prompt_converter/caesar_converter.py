import string
import asyncio

from pyrit.models import PromptDataType
from pyrit.prompt_converter import PromptConverter, ConverterResult


class CaesarConverter(PromptConverter):
    """
    Converter to encode prompt using caesar cipher.

    Encodes by using given offset.
    Using offset=1, 'Hello 123' would encode to 'Ifmmp 234', as each character would shift by 1.
    Shifts for digits 0-9 only work if the offset is less than 10, if the offset is equal to or greather than 10,
    any numeric values will not be shifted.

    Parameters
    ---
    caesar_offset: int, default=3
        Offset for caesar cipher, range 0 to 25 (inclusive). Can also be negative for shifting backwards.
    """

    def __init__(self, *, caesar_offet: int = 3) -> None:
        if caesar_offet < -25 or caesar_offet > 25:
            raise ValueError("caesar offset value invalid, must be between 0 and 25 inclusive.")
        self.caesar_offset = caesar_offet

    async def convert_async(self, *, prompt: str, input_type: PromptDataType = "text") -> ConverterResult:
        """
        Simple converter that caesar cipher encodes the prompt.
        """
        if not self.input_supported(input_type):
            raise ValueError("Input type not supported")

        await asyncio.sleep(0)
        return ConverterResult(output_text=self.caesar(prompt), output_type="text")

    def input_supported(self, input_type: PromptDataType) -> bool:
        return input_type == "text"

    def caesar(self, text: str) -> str:
        def shift(alphabet: str = None) -> str:
            return alphabet[self.caesar_offset :] + alphabet[: self.caesar_offset]

        alphabet = (string.ascii_lowercase, string.ascii_uppercase, string.digits)
        shifted_alphabet = tuple(map(shift, alphabet))
        translation_table = str.maketrans("".join(alphabet), "".join(shifted_alphabet))
        return text.translate(translation_table)
