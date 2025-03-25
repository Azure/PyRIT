# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
from typing import Literal, Optional, Tuple

from pyrit.models import PromptDataType
from pyrit.prompt_converter import ConverterResult, PromptConverter

logger = logging.getLogger(__name__)


class AsciiSmugglerConverter(PromptConverter):
    """
    Converts a string to or from ASCII smuggling encoded format.

    This converter supports two encoding modes:
      - "unicode_tags": (Default) Encodes ASCII characters by mapping them to invisible Unicode tag code points.
          If the `unicode_tags` flag is True, control characters (U+E0001 and U+E007F) are added to mark boundaries.
      - "sneaky_bits": Encodes any text by converting each bit of its UTF-8 representation into one
          of two configurable invisible characters (defaults: U+2062 for binary 0, U+2064 for binary 1).

    Replicates functionality detailed in:
      - https://embracethered.com/blog/posts/2024/hiding-and-finding-text-with-unicode-tags/
      - https://embracethered.com/blog/posts/2025/sneaky-bits-and-ascii-smuggler/
    """

    def __init__(
        self,
        action: Literal["encode", "decode"] = "encode",
        unicode_tags: bool = False,
        encoding_mode: str = "unicode_tags",
        zero_char: Optional[str] = None,
        one_char: Optional[str] = None,
    ):
        """
        Initialize the converter with options for encoding/decoding.

        Args:
            action (Literal["encode", "decode"]): The action to perform.
            unicode_tags (bool): Whether to add Unicode tags (control characters) during Unicode Tags encoding.
            encoding_mode (str): The encoding mode to use; must be either "unicode_tags" (default) or "sneaky_bits".
            zero_char (Optional[str]): Character to represent binary 0 in sneaky_bits mode (default: U+2062).
            one_char (Optional[str]): Character to represent binary 1 in sneaky_bits mode (default: U+2064).

        Raises:
            ValueError: If an unsupported action or encoding_mode is provided.
        """
        self.unicode_tags = unicode_tags
        self.action = action
        self.encoding_mode = encoding_mode
        self.zero_char = zero_char if zero_char is not None else "\u2062"  # Invisible Times
        self.one_char = one_char if one_char is not None else "\u2064"  # Invisible Plus

        if self.action not in ["encode", "decode"]:
            raise ValueError("Action must be either 'encode' or 'decode'")
        if self.encoding_mode not in ["unicode_tags", "sneaky_bits"]:
            raise ValueError("encoding_mode must be either 'unicode_tags' or 'sneaky_bits'")

    async def convert_async(self, *, prompt: str, input_type: PromptDataType = "text") -> ConverterResult:
        """
        Process the prompt to either encode or decode based on options and prompt type.

        Args:
            prompt (str): The prompt to be converted.
            input_type (PromptDataType): The type of input, defaults to "text".

        Returns:
            ConverterResult: The converted result.

        Raises:
            ValueError: If the input type is not supported.
        """
        if not self.input_supported(input_type):
            raise ValueError("Input type not supported")
        # Determine action based on input: either "encode" or "decode"
        if self.action == "encode":
            code_points, encoded = self.encode_message(message=prompt)
            logger.info(f"Encoded message: {code_points}")
            return ConverterResult(output_text=encoded, output_type="text")
        else:  # decode
            # Look into and document the "unicode_escape" decoding
            # input_text = prompt.encode("ascii", "ignore").decode("unicode_escape")
            decoded_message = self.decode_message(message=prompt)
            return ConverterResult(output_text=decoded_message, output_type="text")

    def input_supported(self, input_type: PromptDataType) -> bool:
        return input_type == "text"

    def output_supported(self, output_type: PromptDataType) -> bool:
        return output_type == "text"

    def encode_message(self, *, message: str) -> Tuple[str, str]:
        """
        Encode the message using the selected encoding mode.

        For "unicode_tags" mode, ASCII printable characters (0x20-0x7E) are converted into the
        Unicode Tags block (by adding 0xE0000), with optional wrapping using control tags.
        For "sneaky_bits" mode, the message is first converted to UTF-8 bytes and then each bit is
        represented by the invisible characters (self.zero_char for 0, self.one_char for 1).

        Args:
            message (str): The message to encode.

        Returns:
            Tuple[str, str]: A tuple containing a summary of the code points (or bit info) and the encoded message.
        """
        if self.encoding_mode == "unicode_tags":
            return self._encode_unicode_tags(message)
        elif self.encoding_mode == "sneaky_bits":
            return self._encode_sneaky_bits(message)
        else:
            raise ValueError(f"Unsupported encoding_mode: {self.encoding_mode}")

    def decode_message(self, *, message: str) -> str:
        """
        Decode the message from the selected encoding mode.

        For "unicode_tags" mode, characters in the Unicode Tags block are mapped back to ASCII.
        For "sneaky_bits" mode, the invisible characters are interpreted as bits to reconstruct the
        original UTF-8 byte sequence.

        Args:
            message (str): The encoded message.

        Returns:
            str: The decoded original message.
        """
        if self.encoding_mode == "unicode_tags":
            return self._decode_unicode_tags(message)
        elif self.encoding_mode == "sneaky_bits":
            return self._decode_sneaky_bits(message)
        else:
            raise ValueError(f"Unsupported encoding_mode: {self.encoding_mode}")

    def _encode_unicode_tags(self, message: str) -> Tuple[str, str]:
        """
        Encode the message using Unicode Tag characters.

        Each ASCII printable character (0x20-0x7E) is replaced with a corresponding Unicode Tag
        character (by adding 0xE0000). If the `unicode_tags` flag is True, a leading control tag
        (U+E0001) and a trailing control tag (U+E007F) are added.

        Args:
            message (str): The message to encode.

        Returns:
            Tuple[str, str]: A tuple where the first element is a string summary of code points
            and the second element is the encoded message.
        """
        encoded = ""
        code_points = ""
        invalid_chars = ""
        if self.unicode_tags:
            encoded += chr(0xE0001)
            code_points += "U+E0001 "

        for char in message:
            if 0x20 <= ord(char) <= 0x7E:
                code_point = 0xE0000 + ord(char)
                encoded += chr(code_point)
                code_points += f"U+{code_point:X} "
            else:
                invalid_chars += char

        if self.unicode_tags:
            encoded += chr(0xE007F)
            code_points += "U+E007F"

        if invalid_chars:
            logger.error(f"Invalid characters detected: {invalid_chars}")

        return code_points, encoded

    def _decode_unicode_tags(self, message: str) -> str:
        """
        Decode the message encoded using Unicode Tag characters.

        For each character in the input, if it falls in the Unicode Tag range (0xE0000-0xE007F),
        it is converted back to the corresponding ASCII character (by subtracting 0xE0000). Special
        control tag characters are skipped.

        Args:
            message (str): The encoded message.

        Returns:
            str: The decoded original message.
        """
        decoded_message = ""
        for char in message:
            code_point = ord(char)
            if 0xE0000 <= code_point <= 0xE007F:
                decoded_char = chr(code_point - 0xE0000)
                if not 0x20 <= ord(decoded_char) <= 0x7E:
                    logger.info(f"Potential unicode tag detected: {decoded_char}")
                else:
                    decoded_message += decoded_char
        if len(decoded_message) != len(message):
            logger.info("Hidden Unicode Tags discovered.")
        else:
            logger.info("No hidden Unicode Tag characters discovered.")
        return decoded_message

    def _encode_sneaky_bits(self, message: str) -> Tuple[str, str]:
        """
        Encode the message using Sneaky Bits mode.

        The message is first converted to its UTF-8 byte sequence. Then each byte is represented
        as 8 bits, with each bit replaced by an invisible character (self.zero_char for 0 and
        self.one_char for 1).

        Args:
            message (str): The message to encode.

        Returns:
            Tuple[str, str]: A tuple where the first element is a bit summary (empty in this implementation)
            and the second element is the encoded message containing the invisible bits.
        """
        encoded_bits = []
        data = message.encode("utf-8")
        for byte in data:
            for bit_index in range(7, -1, -1):
                bit = (byte >> bit_index) & 1
                encoded_bits.append(self.zero_char if bit == 0 else self.one_char)
        encoded_text = "".join(encoded_bits)
        # For Sneaky Bits, we do not provide a code point summary.
        bit_summary = ""
        logger.info(
            f"Sneaky Bits encoding complete: {len(message)} characters encoded into {len(encoded_text)} invisible bits."
        )
        return bit_summary, encoded_text

    def _decode_sneaky_bits(self, message: str) -> str:
        """
        Decode the message encoded using Sneaky Bits mode.

        The method filters out only the valid invisible characters (self.zero_char and self.one_char),
        groups them into 8-bit chunks, reconstructs each byte, and finally decodes the byte sequence
        using UTF-8.

        Args:
            message (str): The message encoded with Sneaky Bits.

        Returns:
            str: The decoded original message.
        """
        # Filter only the valid bit characters.
        bit_chars = [c for c in message if c == self.zero_char or c == self.one_char]
        bit_count = len(bit_chars)
        if bit_count % 8 != 0:
            logger.warning("Encoded bit string length is not a multiple of 8. Ignoring incomplete trailing bits.")
            bit_count -= bit_count % 8
            bit_chars = bit_chars[:bit_count]

        decoded_bytes = bytearray()
        current_byte = 0
        bits_collected = 0
        for c in bit_chars:
            current_byte = (current_byte << 1) | (1 if c == self.one_char else 0)
            bits_collected += 1
            if bits_collected == 8:
                decoded_bytes.append(current_byte)
                current_byte = 0
                bits_collected = 0

        try:
            decoded_text = decoded_bytes.decode("utf-8")
        except UnicodeDecodeError:
            decoded_text = decoded_bytes.decode("utf-8", errors="replace")
            logger.error("Decoded byte sequence is not valid UTF-8; some characters may be replaced.")

        logger.info(f"Sneaky Bits decoding complete: Decoded text length is {len(decoded_text)} characters.")
        return decoded_text
