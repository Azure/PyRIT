# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import csv
import json
from typing import Any, Dict, List, Union
import uuid
from datetime import datetime
from pathlib import Path
from collections.abc import MutableMapping

from sqlalchemy.inspection import inspect

from pyrit.memory.memory_models import Base


class MemoryExporter:
    """
    Handles the export of data to various formats, currently supporting only JSON format.
    This class utilizes the strategy design pattern to select the appropriate export format.
    """

    def __init__(self):
        # Using strategy design pattern for export functionality.
        self.export_strategies = {
            "json": self.export_to_json,
            "csv": self.export_to_csv,
            # Future formats can be added here
        }

    def export_data(
        self, data: Union[List[Base], List[Dict]], *, file_path: Path = None, export_type: str = "json"
    ):  # type: ignore
        """
        Exports the provided data to a file in the specified format.

        Args:
            data (Union[List[Base], List[Dict]]): The data to be exported, typically a list of SQLAlchemy
              model instances or as a list of dictionaries.
            file_path (str): The full path, including the file name, where the data will be exported.
            export_type (str, Optional): The format for exporting data. Defaults to "json".

        Raises:
            ValueError: If no file_path is provided or if the specified export format is not supported.
        """
        if not file_path:
            raise ValueError("Please provide a valid file path for exporting data.")

        export_func = self.export_strategies.get(export_type)
        if export_func:
            export_func(data, file_path)
        else:
            raise ValueError(f"Unsupported export format: {export_type}")

    def export_to_json(self, data: Union[List[Base], List[Dict]], file_path: Path = None) -> None:  # type: ignore
        """
        Exports the provided data to a JSON file at the specified file path.
        Each item in the data list, representing a row from the table,
        is converted to a dictionary before being written to the file.

        Args:
            data (Union[List[Base], List[Dict]]): The data to be exported, as a list of SQLAlchemy model instances
              or as a list of dictionaries.
            file_path (Path): The full path, including the file name, where the data will be exported.

        Raises:
            ValueError: If no file_path is provided.
        """
        if not file_path:
            raise ValueError("Please provide a valid file path for exporting data.")

        export_data = [self.model_to_dict(instance) if isinstance(instance, Base) else instance for instance in data]
        with open(file_path, "w") as f:
            json.dump(export_data, f, indent=4)

    def export_to_csv(self, data: Union[List[Base], List[Dict]], file_path: Path = None) -> None:  # type: ignore
        """
        Exports the provided data to a CSV file at the specified file path.
        Each item in the data list, representing a row from the table,
        is converted to a dictionary before being written to the file.

        Args:
            data (Union[List[Base], List[Dict]]): The data to be exported, as a list of SQLAlchemy model instances
              or as a list of dictionaries.
            file_path (Path): The full path, including the file name, where the data will be exported.

        Raises:
            ValueError: If no file_path is provided.
        """
        if not file_path:
            raise ValueError("Please provide a valid file path for exporting data.")
        if not data:
            raise ValueError("No data to export.")

        export_data = [
            _flatten_dict(self.model_to_dict(instance)) if isinstance(instance, Base) else _flatten_dict(instance)
            for instance in data
        ]
        fieldnames = list(export_data[0].keys())
        with open(file_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(export_data)

    def model_to_dict(self, model_instance: Base):  # type: ignore
        """
        Converts an SQLAlchemy model instance into a dictionary, serializing
        special data types such as UUID and datetime to string representations.
        This ensures compatibility with JSON and other serialization formats.

        Args:
            model_instance: An instance of an SQLAlchemy model.

        Returns:
            A dictionary representation of the model instance, with special types serialized.
        """
        model_dict = {}
        for column in inspect(model_instance.__class__).columns:
            value = getattr(model_instance, column.name)
            if isinstance(value, uuid.UUID):
                # Convert UUID to string
                model_dict[column.name] = str(value)
            elif isinstance(value, datetime):
                # Convert datetime to an ISO 8601 formatted string
                model_dict[column.name] = value.isoformat()
            else:
                model_dict[column.name] = value
        return model_dict


def _flatten_dict(d: MutableMapping, parent_key: str = "", sep: str = ".") -> MutableMapping:
    items: list[tuple[Any, Any]] = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, MutableMapping):
            items.extend(_flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)
