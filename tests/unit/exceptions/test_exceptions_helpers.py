# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import pytest

from pyrit.exceptions.exceptions_helpers import (
    extract_json_from_string,
    remove_end_md_json,
    remove_markdown_json,
    remove_start_md_json,
)


@pytest.mark.parametrize(
    "input_str, expected_output",
    [
        ('```json\n{"key": "value"}', '{"key": "value"}'),
        ('`json\n{"key": "value"}', '{"key": "value"}'),
        ('json{"key": "value"}', '{"key": "value"}'),
        ('{"key": "value"}', '{"key": "value"}'),
        ("No JSON here", "No JSON here"),
        ('```jsn\n{"key": "value"}\n```', 'jsn\n{"key": "value"}\n```'),
    ],
)
def test_remove_start_md_json(input_str, expected_output):
    assert remove_start_md_json(input_str) == expected_output


@pytest.mark.parametrize(
    "input_str, expected_output",
    [
        ('{"key": "value"}\n```', '{"key": "value"}'),
        ('{"key": "value"}\n`', '{"key": "value"}'),
        ('{"key": "value"}`', '{"key": "value"}'),
        ('{"key": "value"}', '{"key": "value"}'),
    ],
)
def test_remove_end_md_json(input_str, expected_output):
    assert remove_end_md_json(input_str) == expected_output


@pytest.mark.parametrize(
    "input_str, expected_output",
    [
        ('Some text before JSON {"key": "value"} some text after JSON', '{"key": "value"}'),
        ("Some text before JSON [1, 2, 3] some text after JSON", "[1, 2, 3]"),
        ('{"key": "value"}', '{"key": "value"}'),
        ("[1, 2, 3]", "[1, 2, 3]"),
        ("No JSON here", "No JSON here"),
        ('jsn\n{"key": "value"}\n```', '{"key": "value"}'),
        ('Some text before JSON {"a": [1,2,3], "b": {"c": 4}} some text after JSON', '{"a": [1,2,3], "b": {"c": 4}}'),
    ],
)
def test_extract_json_from_string(input_str, expected_output):
    assert extract_json_from_string(input_str) == expected_output


@pytest.mark.parametrize(
    "input_str, expected_output",
    [
        ('```json\n{"key": "value"}\n```', '{"key": "value"}'),
        ('```json\n{"key": "value"}', '{"key": "value"}'),
        ('{"key": "value"}\n```', '{"key": "value"}'),
        ('Some text before JSON ```json\n{"key": "value"}\n``` some text after JSON', '{"key": "value"}'),
        ('```json\n{"key": "value"\n```', 'Invalid JSON response: {"key": "value"'),
        ("No JSON here", "Invalid JSON response: No JSON here"),
        ('```jsn\n{"key": "value"}\n```', '{"key": "value"}'),
    ],
)
def test_remove_markdown_json(input_str, expected_output):
    assert remove_markdown_json(input_str) == expected_output
