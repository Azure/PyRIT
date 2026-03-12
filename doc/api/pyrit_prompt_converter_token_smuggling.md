# pyrit.prompt_converter.token_smuggling

Token smuggling converters that use Unicode-based techniques to hide, encode,
or obfuscate text content within prompts for security testing purposes.

## `class AsciiSmugglerConverter(SmugglerConverter)`

Implements encoding and decoding using Unicode Tags.

If 'control' is True, the encoded output is wrapped with:
    - U+E0001 (start control tag)
    - U+E007F (end control tag)

Replicates the functionality detailed in the following blog post:
https://embracethered.com/blog/posts/2024/hiding-and-finding-text-with-unicode-tags/

**Constructor Parameters:**

| Parameter | Type | Description |
|---|---|---|
| `action` | `Literal['encode', 'decode']` | The action to perform. Defaults to `'encode'`. |
| `unicode_tags` | `bool` | Whether to add Unicode tags during encoding. Defaults to `False`. |

**Methods:**

#### `decode_message(message: str) → str`

Decode a message encoded with Unicode Tags.

For each character in the Unicode Tags range, subtracts 0xE0000.
Skips control tags if present.

| Parameter | Type | Description |
|---|---|---|
| `message` | `str` | The encoded message. |

**Returns:**

- `str` — The decoded message.

#### `encode_message(message: str) → tuple[str, str]`

Encode the message using Unicode Tags.

Each ASCII printable character (0x20-0x7E) is mapped to a corresponding
Unicode Tag (by adding 0xE0000). If control mode is enabled, wraps the output.

| Parameter | Type | Description |
|---|---|---|
| `message` | `str` | The message to encode. |

**Returns:**

- `tuple[str, str]` — Tuple[str, str]: A tuple with a summary of code points and the encoded message.

## `class SneakyBitsSmugglerConverter(SmugglerConverter)`

Encodes and decodes text using a bit-level approach.

**Constructor Parameters:**

| Parameter | Type | Description |
|---|---|---|
| `action` | `Literal['encode', 'decode']` | The action to perform. Defaults to `'encode'`. |
| `zero_char` | `Optional[str]` | Character to represent binary 0 in ``sneaky_bits`` mode (default: U+2062). Defaults to `None`. |
| `one_char` | `Optional[str]` | Character to represent binary 1 in ``sneaky_bits`` mode (default: U+2064). Defaults to `None`. |

**Methods:**

#### `decode_message(message: str) → str`

Decode the message encoded using Sneaky Bits mode.

The method filters out only the valid invisible characters (``self.zero_char`` and ``self.one_char``),
groups them into 8-bit chunks, reconstructs each byte, and finally decodes the byte sequence using UTF-8.

| Parameter | Type | Description |
|---|---|---|
| `message` | `str` | The message encoded with Sneaky Bits. |

**Returns:**

- `str` — The decoded original message.

#### `encode_message(message: str) → tuple[str, str]`

Encode the message using Sneaky Bits mode.

The message is first converted to its UTF-8 byte sequence. Then each byte is represented as 8 bits,
with each bit replaced by an invisible character (``self.zero_char`` for 0 and ``self.one_char`` for 1).

| Parameter | Type | Description |
|---|---|---|
| `message` | `str` | The message to encode. |

**Returns:**

- `str` — Tuple[str, str]: A tuple where the first element is a bit summary (empty in this implementation)
- `str` — and the second element is the encoded message containing the invisible bits.

## `class VariationSelectorSmugglerConverter(SmugglerConverter)`

Extension: In addition to embedding into a base character, we also support
appending invisible variation selectors directly to visible text—enabling mixed
visible and hidden content within a single string.

**Constructor Parameters:**

| Parameter | Type | Description |
|---|---|---|
| `action` | `Literal['encode', 'decode']` | The action to perform. Defaults to `'encode'`. |
| `base_char_utf8` | `Optional[str]` | Base character for ``variation_selector_smuggler`` mode (default: 😊). Defaults to `None`. |
| `embed_in_base` | `bool` | If True, the hidden payload is embedded directly into the base character.                     If False, a visible separator (space) is inserted between the base and payload.                     Default is True. Defaults to `True`. |

**Methods:**

#### `decode_message(message: str) → str`

Decode a message encoded using Unicode variation selectors.
The decoder scans the string for variation selectors, ignoring any visible separator.

| Parameter | Type | Description |
|---|---|---|
| `message` | `str` | The encoded message. |

**Returns:**

- `str` — The decoded message.

#### `decode_visible_hidden(combined: str) → tuple[str, str]`

Extract the visible text and decodes the hidden text from a combined string.

It searches for the first occurrence of the base character (``self.utf8_base_char``) and treats everything
from that point on as the hidden payload.

| Parameter | Type | Description |
|---|---|---|
| `combined` | `str` | The combined text containing visible and hidden parts. |

**Returns:**

- `tuple[str, str]` — Tuple[str, str]: A tuple with the visible text and the decoded hidden text.

#### `encode_message(message: str) → tuple[str, str]`

Encode the message using Unicode variation selectors.

The message is converted to UTF-8 bytes, and each byte is mapped to a variation selector:
    - 0x00-0x0F => U+FE00 to U+FE0F.
    - 0x10-0xFF => U+E0100 to U+E01EF.

If ``embed_in_base`` is True, the payload is embedded directly into the base character;
otherwise, a visible separator (a space) is inserted between the base and payload.

| Parameter | Type | Description |
|---|---|---|
| `message` | `str` | The message to encode. |

**Returns:**

- `tuple[str, str]` — Tuple[str, str]: A tuple containing a summary of the code points and the encoded string.

#### `encode_visible_hidden(visible: str, hidden: str) → tuple[str, str]`

Combine visible text with hidden text by encoding the hidden text using ``variation_selector_smuggler`` mode.

The hidden payload is generated as a composite using the current embedding setting and then appended
to the visible text.

| Parameter | Type | Description |
|---|---|---|
| `visible` | `str` | The visible text. |
| `hidden` | `str` | The secret/hidden text to encode. |

**Returns:**

- `tuple[str, str]` — Tuple[str, str]: A tuple containing a summary and the combined text.
