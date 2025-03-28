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

    This converter supports three encoding modes:
      - "unicode_tags": Encodes ASCII characters using the Unicode Tags block, without control tag boundaries.
      - "unicode_tags_control": Same as "unicode_tags", but wraps the output with control characters.
      - "sneaky_bits": Encodes UTF-8 data at the bit level using two invisible Unicode characters.
      - "variation_selector_smuggler": Encodes arbitrary UTF-8 data using Unicode variation selectors.

    Replicates functionality detailed in:
      - https://embracethered.com/blog/posts/2024/hiding-and-finding-text-with-unicode-tags/
      - https://embracethered.com/blog/posts/2025/sneaky-bits-and-ascii-smuggler/
      - https://paulbutler.org/2025/smuggling-arbitrary-data-through-an-emoji/
    """

    def __init__(
        self,
        action: Literal["encode", "decode"] = "encode",
        encoding_mode: Literal[
            "unicode_tags", "unicode_tags_control", "sneaky_bits", "variation_selector_smuggler"
        ] = "unicode_tags",
        zero_char: Optional[str] = None,
        one_char: Optional[str] = None,
        base_char_utf8: Optional[str] = None,
        embed_in_base: bool = True,
    ):
        """
        Initialize the converter with options for encoding/decoding.

        Args:
            action (Literal["encode", "decode"]): The action to perform.
            encoding_mode (Literal["unicode_tags", "unicode_tags_control", "sneaky_bits",
            "variation_selector_smuggler"]): Encoding mode to use.
            zero_char (Optional[str]): Character to represent binary 0 in sneaky_bits mode (default: U+2062).
            one_char (Optional[str]): Character to represent binary 1 in sneaky_bits mode (default: U+2064).
            base_char_utf8 (Optional[str]): Base character for variation_selector_smuggler mode (default: 😊).
            embed_in_base (bool): If True, the hidden payload is embedded directly into the base character.
                                    If False, a visible separator (space) is inserted between the base and payload.
                                    Default is True.

        Raises:
            ValueError: If an unsupported action or encoding_mode is provided.
        """
        if action not in ["encode", "decode"]:
            raise ValueError("Action must be either 'encode' or 'decode'")
        if encoding_mode not in ["unicode_tags", "unicode_tags_control", "sneaky_bits", "variation_selector_smuggler"]:
            raise ValueError(
                "encoding_mode must be one of: 'unicode_tags', 'unicode_tags_control', "
                "'sneaky_bits', 'variation_selector_smuggler'"
            )

        self.action = action
        self.encoding_mode = encoding_mode
        self.zero_char = zero_char if zero_char is not None else "\u2062"  # Invisible Times
        self.one_char = one_char if one_char is not None else "\u2064"  # Invisible Plus
        self.utf8_base_char = base_char_utf8 if base_char_utf8 is not None else "😊"
        self.embed_in_base = embed_in_base

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
        For "variation_selector_smuggler" mode, the message is converted to UTF-8 bytes
        and each byte is encoded as a variation selector:
          - 0x00-0x0F => U+FE00 to U+FE0F.
          - 0x10-0xFF => U+E0100 to U+E01EF.
        In variation_selector_smuggler mode, if embed_in_base is True (default),
        the hidden payload is embedded directly into the base character; otherwise, a visible separator is inserted.

        Args:
            message (str): The message to encode.

        Returns:
            Tuple[str, str]: A tuple containing a summary of the code points (or bit info) and the encoded message.
        """
        if self.encoding_mode in ["unicode_tags", "unicode_tags_control"]:
            return self._encode_unicode_tags(message)
        elif self.encoding_mode == "sneaky_bits":
            return self._encode_sneaky_bits(message)
        elif self.encoding_mode == "variation_selector_smuggler":
            return self._encode_variation_selector_smuggler(message)
        else:
            raise ValueError(f"Unsupported encoding_mode: {self.encoding_mode}")

    def decode_message(self, *, message: str) -> str:
        """
        Decode the message from the selected encoding mode.

        For "unicode_tags" mode, characters in the Unicode Tags block are mapped back to ASCII.
        For "sneaky_bits" mode, the invisible characters are interpreted as bits to reconstruct the
        original UTF-8 byte sequence.
        For "variation_selector_smuggler" mode, variation selectors (ignoring an initial base character) are mapped
        back to their corresponding UTF-8 bytes.

        Args:
            message (str): The encoded message.

        Returns:
            str: The decoded original message.
        """
        if self.encoding_mode in ["unicode_tags", "unicode_tags_control"]:
            return self._decode_unicode_tags(message)
        elif self.encoding_mode == "sneaky_bits":
            return self._decode_sneaky_bits(message)
        elif self.encoding_mode == "variation_selector_smuggler":
            return self._decode_variation_selector_smuggler(message)
        else:
            raise ValueError(f"Unsupported encoding_mode: {self.encoding_mode}")

    def _encode_unicode_tags(self, message: str) -> Tuple[str, str]:
        """
        Encode the message using Unicode Tag characters.

        Each ASCII printable character (0x20-0x7E) is replaced with a corresponding Unicode Tag
        character (by adding 0xE0000). In "unicode_tags_control" mode, a leading control tag
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

        if self.encoding_mode == "unicode_tags_control":
            encoded += chr(0xE0001)
            code_points += "U+E0001 "

        for char in message:
            if 0x20 <= ord(char) <= 0x7E:
                code_point = 0xE0000 + ord(char)
                encoded += chr(code_point)
                code_points += f"U+{code_point:X} "
            else:
                invalid_chars += char

        if self.encoding_mode == "unicode_tags_control":
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

            # Explicitly skip control boundary tags for unicode_tags_control
            if self.encoding_mode == "unicode_tags_control" and code_point in (0xE0001, 0xE007F):
                continue

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

    def _encode_variation_selector_smuggler(self, message: str) -> Tuple[str, str]:
        """
        Encode the message using Unicode variation selectors.
        The message is converted to UTF-8 bytes, and each byte is mapped to a variation selector:
          - 0x00-0x0F => U+FE00 to U+FE0F.
          - 0x10-0xFF => U+E0100 to U+E01EF.
        If embed_in_base is True, the payload is embedded directly into the base character;
        otherwise, a visible separator (a space) is inserted between the base and payload.
        """
        base_char = self.utf8_base_char
        payload = ""
        data = message.encode("utf-8")
        for byte in data:
            if byte < 16:
                code_point = 0xFE00 + byte
            else:
                code_point = 0xE0100 + (byte - 16)
            payload += chr(code_point)
        if self.embed_in_base:
            encoded = base_char + payload
        else:
            encoded = base_char + " " + payload
        summary_parts = [f"Base char: U+{ord(base_char):X}"]
        for byte in data:
            if byte < 16:
                summary_parts.append(f"U+{(0xFE00 + byte):X}")
            else:
                summary_parts.append(f"U+{(0xE0100 + (byte - 16)):X}")
        code_points_summary = " ".join(summary_parts)
        logger.info(f"Variation Selector Smuggler encoding complete: {len(data)} bytes encoded.")
        return code_points_summary.strip(), encoded

    def _decode_variation_selector_smuggler(self, message: str) -> str:
        """
        Decode a message encoded using Unicode variation selectors.
        The decoder scans the string for variation selectors, ignoring any visible separator.
        """
        bytes_out = bytearray()
        started = False
        for char in message:
            # If not embedding, skip visible separators (e.g., spaces)
            if not self.embed_in_base and char == " ":
                continue
            code = ord(char)
            if 0xFE00 <= code <= 0xFE0F:
                started = True
                byte = code - 0xFE00
                bytes_out.append(byte)
            elif 0xE0100 <= code <= 0xE01EF:
                started = True
                byte = (code - 0xE0100) + 16
                bytes_out.append(byte)
            else:
                if started:
                    break
        try:
            decoded_text = bytes_out.decode("utf-8")
        except UnicodeDecodeError:
            decoded_text = bytes_out.decode("utf-8", errors="replace")
            logger.error("Decoded byte sequence is not valid UTF-8; some characters may be replaced.")
        logger.info(f"Variation Selector Smuggler decoding complete: {len(decoded_text)} characters decoded.")
        return decoded_text

    # New methods for handling visible and hidden text together.
    def encode_visible_hidden(self, visible: str, hidden: str) -> Tuple[str, str]:
        """
        Combine visible text with hidden text by encoding the hidden text using variation_selector_smuggler mode.
        The hidden payload is generated as a composite using the current embedding setting and then appended
        to the visible text.
        Args:
            visible (str): The visible text.
            hidden (str): The secret/hidden text to encode.
        Returns:
            Tuple[str, str]: A tuple containing a summary and the combined text.
        """
        summary, encoded_hidden = self._encode_variation_selector_smuggler(hidden)
        combined = visible + encoded_hidden
        return summary, combined

    def decode_visible_hidden(self, combined: str) -> Tuple[str, str]:
        """
        Extract the visible text and decode the hidden text from a combined string.
        It searches for the first occurrence of the base character (self.utf8_base_char) and treats everything
        from that point on as the hidden payload.
        Args:
            combined (str): The combined text containing visible and hidden parts.
        Returns:
            Tuple[str, str]: A tuple with the visible text and the decoded hidden text.
        """
        base_char = self.utf8_base_char
        index = combined.find(base_char)
        if index == -1:
            return combined, ""
        visible = combined[:index]
        hidden_encoded = combined[index:]
        hidden = self._decode_variation_selector_smuggler(hidden_encoded)
        return visible, hidden
