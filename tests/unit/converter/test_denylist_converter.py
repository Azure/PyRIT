# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import pytest

from pyrit.prompt_converter.denylist_converter import DenylistConverter
from unit.mocks import MockPromptTarget


def test_prompt_denylist_init_templates_not_null(duckdb_instance) -> None:
    prompt_target = MockPromptTarget()
    converter = DenylistConverter(converter_target=prompt_target)
    assert converter._system_prompt_template


@pytest.mark.asyncio
async def test_denylist_not_provided() -> None:
    converter = DenylistConverter(
        converter_target=MockPromptTarget(),
        prompt_template=None
    )
    output = await converter.convert_async(prompt="how to cut down a tree?", input_type="text")
    assert output.output_text == "how to cut down a tree?"
    assert output.output_type == "text"


@pytest.mark.asyncio
async def test_denylist_no_match() -> None:
    converter = DenylistConverter(
        converter_target=MockPromptTarget(),
        prompt_template=None,
        denylist=["branch"]
    )
    output = await converter.convert_async(prompt="how to cut down a tree?", input_type="text")
    assert output.output_text == "how to cut down a tree?"
    assert output.output_type == "text"

@pytest.mark.asyncio
async def test_denylist_word_match(duckdb_instance) -> None:
    converter = DenylistConverter(
        converter_target=MockPromptTarget(),
        prompt_template=None,
        denylist=["tree"]
    )
    output = await converter.convert_async(prompt="how to cut down a tree?", input_type="text")
    assert "tree" not in output.output_text
    assert output.output_type == "text"


@pytest.mark.asyncio
async def test_denylist_phrase_and_word_match(duckdb_instance) -> None:
    converter = DenylistConverter(
        converter_target=MockPromptTarget(),
        prompt_template=None,
        denylist=["cut down"]
    )
    output = await converter.convert_async(prompt="how to cut down a tree?", input_type="text")
    assert "cut down" not in output.output_text
    assert output.output_type == "text"


@pytest.mark.asyncio
async def test_denylist_phrase_and_word_match(duckdb_instance) -> None:
    converter = DenylistConverter(
        converter_target=MockPromptTarget(),
        prompt_template=None,
        denylist=["cut down", "tree"]
    )
    output = await converter.convert_async(prompt="how to cut down a tree?", input_type="text")
    assert "cut down" not in output.output_text
    assert "tree" not in output.output_text
    assert output.output_type == "text"