# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
import warnings
from pathlib import Path
from typing import Optional, TypeVar, Union

logger = logging.getLogger(__name__)

Model = TypeVar("Model")


class DuckDBMemory:
    """
    A class to manage conversation memory using DuckDB as the backend database.

    .. warning::
        DuckDBMemory has been replaced by SQLiteMemory for better compatibility and performance.
        This class will raise a NotImplementedError when instantiated.
        Please use SQLiteMemory instead.
    """

    DEFAULT_DB_FILE_NAME = "pyrit_duckdb_storage.db"

    def __init__(
        self,
        *,
        db_path: Optional[Union[Path, str]] = None,
        verbose: bool = False,
    ):
        # Issue warning and prevent usage
        warnings.warn(
            "DuckDBMemory has been replaced by SQLiteMemory for better compatibility and performance. "
            "Please use SQLiteMemory instead. DuckDBMemory will not function.",
            UserWarning,
            stacklevel=2,
        )
        raise NotImplementedError(
            "DuckDBMemory has been replaced by SQLiteMemory. "
            "Please use SQLiteMemory instead for better compatibility and performance."
        )
