# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

from pyrit.prompt_target.openai.openai_error_handling import (
    _is_content_filter_error,
)

# Tests for _is_content_filter_error helper


def test_is_content_filter_error_with_dict():
    """Test detection with dict input"""
    data = {"error": {"code": "content_filter"}}
    assert _is_content_filter_error(data) is True


def test_is_content_filter_error_with_string():
    """Test detection with string input containing content_filter"""
    error_str = '{"error": {"code": "content_filter"}}'
    assert _is_content_filter_error(error_str) is True


def test_is_content_filter_error_no_filter():
    """Test detection returns False when no content_filter"""
    error_dict = {"error": {"code": "rate_limit", "message": "Too many requests"}}
    assert _is_content_filter_error(error_dict) is False
