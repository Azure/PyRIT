# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class JsonResponseConfig:
    """
    Configuration for JSON responses (with OpenAI).
    """

    enabled: bool = False
    schema: Optional[Dict[str, Any]] = None
    schema_name: str = "CustomSchema"
    strict: bool = True

    @classmethod
    def from_metadata(cls, *, metadata: Optional[Dict[str, Any]]) -> JsonResponseConfig:
        if not metadata:
            return cls(enabled=False)

        response_format = metadata.get("response_format")
        if response_format != "json":
            return cls(enabled=False)

        schema_val = metadata.get("json_schema")
        if schema_val:
            if isinstance(schema_val, str):
                try:
                    schema = json.loads(schema_val) if schema_val else None
                except json.JSONDecodeError:
                    raise ValueError(f"Invalid JSON schema provided: {schema_val}")
            else:
                schema = schema_val

            return cls(
                enabled=True,
                schema=schema,
                schema_name=metadata.get("schema_name", "CustomSchema"),
                strict=metadata.get("strict", True),
            )

        return cls(enabled=True)
