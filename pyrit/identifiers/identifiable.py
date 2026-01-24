# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

import hashlib
import json
from abc import abstractmethod
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from typing import Any, Dict, List, Literal, Optional, Type, TypeVar

from pyrit.common.deprecation import print_deprecation_message
from pyrit.identifiers.class_name_utils import class_name_to_snake_case

class LegacyIdentifiable:
    """
    Deprecated legacy interface for objects that can provide an identifier dictionary.

    This interface will eventually be replaced by Identifier dataclass.
    Classes implementing this interface should return a dict describing their identity.
    """

    @abstractmethod
    def get_identifier(self) -> dict[str, str]:
        pass

    def __str__(self) -> str:
        return f"{self.get_identifier}"

class Identifiable(LegacyIdentifiable):
    """
    Abstract base class for objects that can provide an identifier dictionary.

    This is a legacy interface that will eventually be replaced by Identifier dataclass.
    Classes implementing this interface should return a dict describing their identity.
    """

    @abstractmethod
    def get_identifier(self) -> dict[str, str]:
        pass

    def __str__(self) -> str:
        return f"{self.get_identifier}"
    

