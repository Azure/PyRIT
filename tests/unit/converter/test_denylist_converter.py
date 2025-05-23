# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
from unittest.mock import AsyncMock, MagicMock

import pytest
from unit.mocks import MockPromptTarget

from pyrit.models.prompt_request_piece import PromptRequestPiece
from pyrit.models.prompt_request_response import PromptRequestResponse
from pyrit.models.seed_prompt import SeedPrompt
from pyrit.prompt_converter.denylist_converter import DenylistConverter


@pytest.fixture
def mock_template():
    mock = MagicMock(spec=SeedPrompt)
    mock.render_template_value = MagicMock(return_value="mocked system prompt")
    return mock


@pytest.fixture
def mock_target() -> MockPromptTarget:
    target = MagicMock()
    response = PromptRequestResponse(
        request_pieces=[
            PromptRequestPiece(
                role="assistant",
                original_value="prompt value",
            )
        ]
    )
    target.send_prompt_async = AsyncMock(return_value=response)
    return target


def test_prompt_denylist_init_templates_not_null(duckdb_instance) -> None:
    prompt_target = MockPromptTarget()
    converter = DenylistConverter(converter_target=prompt_target)
    assert converter._system_prompt_template


def test_prompt_denylist_init_template_provided(duckdb_instance, mock_template) -> None:
    prompt_target = MockPromptTarget()
    converter = DenylistConverter(converter_target=prompt_target, system_prompt_template=mock_template)
    assert converter._system_prompt_template == mock_template


@pytest.mark.asyncio
async def test_denylist_not_provided() -> None:
    converter = DenylistConverter(converter_target=MockPromptTarget(), system_prompt_template=None)
    output = await converter.convert_async(prompt="how to cut down a tree?", input_type="text")
    assert output.output_text == "how to cut down a tree?"
    assert output.output_type == "text"


@pytest.mark.asyncio
async def test_denylist_no_match() -> None:
    converter = DenylistConverter(converter_target=MockPromptTarget(), system_prompt_template=None, denylist=["branch"])
    output = await converter.convert_async(prompt="how to cut down a tree?", input_type="text")
    assert output.output_text == "how to cut down a tree?"
    assert output.output_type == "text"


@pytest.mark.asyncio
async def test_denylist_word_match(duckdb_instance, mock_target) -> None:
    converter = DenylistConverter(converter_target=mock_target, system_prompt_template=None, denylist=["tree"])
    output = await converter.convert_async(prompt="how to cut down a tree?", input_type="text")
    mock_target.send_prompt_async.assert_called_once()
    assert output.output_type == "text"


@pytest.mark.asyncio
async def test_denylist_phrase_match(duckdb_instance, mock_target) -> None:
    converter = DenylistConverter(converter_target=mock_target, system_prompt_template=None, denylist=["cut down"])
    output = await converter.convert_async(prompt="how to cut down a tree?", input_type="text")
    mock_target.send_prompt_async.assert_called_once()
    assert output.output_type == "text"


@pytest.mark.asyncio
async def test_denylist_phrase_and_word_match(duckdb_instance) -> None:
    converter = DenylistConverter(
        converter_target=MockPromptTarget(), system_prompt_template=None, denylist=["cut down", "tree"]
    )
    output = await converter.convert_async(prompt="how to cut down a tree?", input_type="text")
    assert "cut down" not in output.output_text
    assert "tree" not in output.output_text
    assert output.output_type == "text"
