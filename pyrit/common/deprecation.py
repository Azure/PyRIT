# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

from typing import Any, Callable


def deprecation_message(
    *,
    old_item: type | Callable[..., Any] | str,
    new_item: type | Callable[..., Any] | str,
    removed_in: str,
) -> str:
    """
    Generate a deprecation message string.

    Args:
        old_item: The deprecated class, function, or its string name
        new_item: The replacement class, function, or its string name
        removed_in: The version in which the deprecated item will be removed

    Returns:
        A formatted deprecation message string
    """
    # Get the qualified name for old item
    if callable(old_item) or isinstance(old_item, type):
        old_name = f"{old_item.__module__}.{old_item.__qualname__}"
    else:
        old_name = old_item

    # Get the qualified name for new item
    if callable(new_item) or isinstance(new_item, type):
        new_name = f"{new_item.__module__}.{new_item.__qualname__}"
    else:
        new_name = new_item

    return f"{old_name} is deprecated and will be removed in {removed_in}; use {new_name} instead."