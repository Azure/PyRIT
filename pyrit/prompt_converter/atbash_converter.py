import string
import asyncio

from pyrit.models import PromptDataType
from pyrit.prompt_converter import PromptConverter, ConverterResult


class AtbashConverter(PromptConverter):
    """
    Converter to encode prompt using atbash cipher.

    Uses the following to encode:
    ABCDEFGHIJKLMNOPQRSTUVWXYZ 0123456789
    ZYXWVUTSRQPONMLKJIHGFEDCBA 9876543210

    'Hello 123' would encode to 'Svool 876'
    """

    async def convert_async(self, *, prompt: str, input_type: PromptDataType = "text") -> ConverterResult:
        """
        Simple converter that atbash cipher encodes the prompt.
        """
        if not self.input_supported(input_type):
            raise ValueError("Input type not supported")

        await asyncio.sleep(0)
        return ConverterResult(output_text=self.atbash(prompt), output_type="text")

    def input_supported(self, input_type: PromptDataType) -> bool:
        return input_type == "text"

    def atbash(self, text: str) -> str:
        def shift(alphabet: str = None) -> str:
            return alphabet[::-1]

        alphabet = (string.ascii_lowercase, string.ascii_uppercase, string.digits)
        shifted_alphabet = tuple(map(shift, alphabet))
        translation_table = str.maketrans("".join(alphabet), "".join(shifted_alphabet))
        return text.translate(translation_table)
