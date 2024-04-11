# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import json
from pyrit.prompt_converter import (
    TxtFileConverter,
)


def test_txt_file_converter_twice() -> None:
    converter = TxtFileConverter()
    result1 = json.loads(converter.convert(prompt="test", input_type="text"))
    with open(result1["file_location"], "r") as txt_file:
        assert txt_file.read() == result1["data"] == "test"

    result2 = json.loads(converter.convert(prompt="test", input_type="text"))
    with open(result2["file_location"], "r") as txt_file:
        assert txt_file.read() == result2["data"] == "test"

    assert result1["file_location"] != result2["file_location"]
