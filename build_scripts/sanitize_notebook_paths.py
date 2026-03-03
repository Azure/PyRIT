# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import json
import re
import sys

# Matches Windows and Unix user home directory prefixes
_USER_PATH_PATTERNS = [
    re.compile(r"[A-Z]:\\+Users\\+[^\\]+\\+"),  # C:\Users\username\ or C:\\Users\\username\\
    re.compile(r"/Users/[^/]+/"),  # /Users/username/ (macOS)
    re.compile(r"/home/[^/]+/"),  # /home/username/ (Linux)
]


def sanitize_notebook_paths(file_path: str) -> bool:
    """
    Remove user-specific path prefixes from notebook cell outputs.

    Args:
        file_path (str): Path to the .ipynb file.

    Returns:
        bool: True if the file was modified.
    """
    if not file_path.endswith(".ipynb"):
        return False

    with open(file_path, encoding="utf-8") as f:
        content = json.load(f)

    original = json.dumps(content)

    for cell in content.get("cells", []):
        for output in cell.get("outputs", []):
            _sanitize_output_field(output, "text")
            _sanitize_output_field(output, "traceback")
            _sanitize_output_field(output, "evalue")
            if "data" in output:
                for mime_type in output["data"]:
                    if mime_type.startswith("text/") or mime_type == "application/json":
                        _sanitize_output_field(output["data"], mime_type)

    modified = json.dumps(content)
    if modified != original:
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(content, f, indent=1)
        return True

    return False


def _sanitize_output_field(obj: dict, key: str) -> None:
    """
    Sanitize a single output field by stripping user path prefixes.

    Args:
        obj (dict): The dict containing the field.
        key (str): The key to sanitize.
    """
    value = obj.get(key)
    if value is None:
        return

    if isinstance(value, list):
        obj[key] = [_strip_user_paths(line) if isinstance(line, str) else line for line in value]
    elif isinstance(value, str):
        obj[key] = _strip_user_paths(value)


def _strip_user_paths(text: str) -> str:
    """
    Strip user-specific path prefixes from a string.

    Args:
        text (str): The text to sanitize.

    Returns:
        str: The sanitized text.
    """
    for pattern in _USER_PATH_PATTERNS:
        text = pattern.sub("", text)
    return text


if __name__ == "__main__":
    modified_files = []
    for file_path in sys.argv[1:]:
        if sanitize_notebook_paths(file_path):
            modified_files.append(file_path)
    if modified_files:
        print("Sanitized user paths in:", modified_files)
        sys.exit(1)
