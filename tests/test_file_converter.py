# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import json
from pyrit.prompt_converter import (
    TxtFileConverter,
)


def test_txt_file_converter_no_file_name() -> None:
    converter = TxtFileConverter()
    result = json.loads(converter.convert(prompt="test", input_type="text"))
    with open(result["file_location"], "r") as txt_file:
        assert txt_file.read() == result["data"] == "test"


def test_txt_file_converter_file_name() -> None:
    converter = TxtFileConverter(file_name="test_file")
    result = json.loads(converter.convert(prompt="test", input_type="text"))
    assert result["file_location"] == "test_file.txt"
    with open(result["file_location"], "r") as txt_file:
        assert txt_file.read() == result["data"] == "test"
