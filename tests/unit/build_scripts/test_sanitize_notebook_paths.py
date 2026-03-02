# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import json
import os
import tempfile

import pytest

from build_scripts.sanitize_notebook_paths import _strip_user_paths, sanitize_notebook_paths


class TestStripUserPaths:
    def test_windows_path(self) -> None:
        assert _strip_user_paths(r"C:\Users\romanlutz\git\PyRIT\foo.py") == r"git\PyRIT\foo.py"

    def test_windows_double_backslash(self) -> None:
        result = _strip_user_paths("C:\\\\Users\\\\alice\\\\AppData\\\\Local\\\\Temp\\\\file.py")
        assert result == "AppData\\\\Local\\\\Temp\\\\file.py"

    def test_linux_path(self) -> None:
        assert _strip_user_paths("/home/user1/projects/pyrit/foo.py") == "projects/pyrit/foo.py"

    def test_macos_path(self) -> None:
        assert _strip_user_paths("/Users/alice/Documents/test.txt") == "Documents/test.txt"

    def test_no_user_path(self) -> None:
        text = "just a normal string"
        assert _strip_user_paths(text) == text

    def test_drive_letter_d(self) -> None:
        assert _strip_user_paths(r"D:\Users\testuser\hello.py") == "hello.py"


class TestSanitizeNotebookPaths:
    def _make_notebook(self, outputs: list) -> dict:
        return {
            "cells": [
                {
                    "cell_type": "code",
                    "outputs": outputs,
                }
            ]
        }

    def test_sanitizes_stderr_output(self) -> None:
        nb = self._make_notebook(
            [
                {
                    "output_type": "stream",
                    "name": "stderr",
                    "text": [r"C:\Users\test\AppData\Local\Temp\file.py:10: Warning"],
                }
            ]
        )
        with tempfile.NamedTemporaryFile(mode="w", suffix=".ipynb", delete=False, encoding="utf-8") as f:
            json.dump(nb, f)
            f_path = f.name
        try:
            assert sanitize_notebook_paths(f_path) is True
            with open(f_path, encoding="utf-8") as f:
                result = json.load(f)
            text = result["cells"][0]["outputs"][0]["text"][0]
            assert "Users" not in text
            assert r"AppData\Local\Temp\file.py:10: Warning" in text
        finally:
            os.unlink(f_path)

    def test_idempotent(self) -> None:
        nb = self._make_notebook(
            [
                {
                    "output_type": "stream",
                    "name": "stderr",
                    "text": [r"C:\Users\test\file.py:10: Warning"],
                }
            ]
        )
        with tempfile.NamedTemporaryFile(mode="w", suffix=".ipynb", delete=False, encoding="utf-8") as f:
            json.dump(nb, f)
            f_path = f.name
        try:
            assert sanitize_notebook_paths(f_path) is True
            assert sanitize_notebook_paths(f_path) is False
        finally:
            os.unlink(f_path)

    def test_skips_non_ipynb(self) -> None:
        assert sanitize_notebook_paths("test.py") is False

    def test_no_modification_when_clean(self) -> None:
        nb = self._make_notebook(
            [
                {
                    "output_type": "stream",
                    "name": "stdout",
                    "text": ["Hello world\n"],
                }
            ]
        )
        with tempfile.NamedTemporaryFile(mode="w", suffix=".ipynb", delete=False, encoding="utf-8") as f:
            json.dump(nb, f)
            f_path = f.name
        try:
            assert sanitize_notebook_paths(f_path) is False
        finally:
            os.unlink(f_path)

    @pytest.mark.parametrize(
        "path,expected_clean",
        [
            (r"C:\Users\alice\project\main.py", r"project\main.py"),
            ("/home/bob/src/test.py", "src/test.py"),
            ("/Users/charlie/docs/readme.md", "docs/readme.md"),
        ],
    )
    def test_various_platforms(self, path: str, expected_clean: str) -> None:
        nb = self._make_notebook([{"output_type": "stream", "name": "stderr", "text": [path]}])
        with tempfile.NamedTemporaryFile(mode="w", suffix=".ipynb", delete=False, encoding="utf-8") as f:
            json.dump(nb, f)
            f_path = f.name
        try:
            sanitize_notebook_paths(f_path)
            with open(f_path, encoding="utf-8") as f:
                result = json.load(f)
            assert result["cells"][0]["outputs"][0]["text"][0] == expected_clean
        finally:
            os.unlink(f_path)
