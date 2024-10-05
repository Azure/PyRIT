# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


from pyrit.exceptions.exception_classes import (
    extract_json_from_string,
    remove_end_md_json,
    remove_markdown_json,
    remove_start_md_json,
)


def test_remove_start_md_json():
    assert remove_start_md_json('```json\n{"key": "value"}') == '{"key": "value"}'
    assert remove_start_md_json('`json\n{"key": "value"}') == '{"key": "value"}'
    assert remove_start_md_json('json{"key": "value"}') == '{"key": "value"}'
    assert remove_start_md_json('{"key": "value"}') == '{"key": "value"}'
    assert remove_start_md_json("No JSON here") == "No JSON here"
    assert remove_start_md_json('```jsn\n{"key": "value"}\n```') == 'jsn\n{"key": "value"}\n```'


def test_remove_end_md_json():
    assert remove_end_md_json('{"key": "value"}\n```') == '{"key": "value"}'
    assert remove_end_md_json('{"key": "value"}\n`') == '{"key": "value"}'
    assert remove_end_md_json('{"key": "value"}`') == '{"key": "value"}'
    assert remove_end_md_json('{"key": "value"}') == '{"key": "value"}'


def test_extract_json_from_string():
    assert extract_json_from_string('Some text before JSON {"key": "value"} some text after JSON') == '{"key": "value"}'
    assert extract_json_from_string("Some text before JSON [1, 2, 3] some text after JSON") == "[1, 2, 3]"
    assert extract_json_from_string('{"key": "value"}') == '{"key": "value"}'
    assert extract_json_from_string("[1, 2, 3]") == "[1, 2, 3]"
    assert extract_json_from_string("No JSON here") == "No JSON here"
    assert extract_json_from_string('jsn\n{"key": "value"}\n```') == '{"key": "value"}'


def test_remove_markdown_json():
    assert remove_markdown_json('```json\n{"key": "value"}\n```') == '{"key": "value"}'
    assert remove_markdown_json('```json\n{"key": "value"}') == '{"key": "value"}'
    assert remove_markdown_json('{"key": "value"}\n```') == '{"key": "value"}'
    assert (
        remove_markdown_json('Some text before JSON ```json\n{"key": "value"}\n``` some text after JSON')
        == '{"key": "value"}'
    )
    assert remove_markdown_json('```json\n{"key": "value"\n```') == 'Invalid JSON response: {"key": "value"'
    assert remove_markdown_json("No JSON here") == "Invalid JSON response: No JSON here"
    assert remove_markdown_json('```jsn\n{"key": "value"}\n```') == '{"key": "value"}'
