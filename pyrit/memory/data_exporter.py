# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import json
from pathlib import Path
from typing import Union
import uuid
from datetime import datetime

from sqlalchemy.inspection import inspect

from pyrit.memory.memory_models import Base
from pyrit.memory.memory_interface import MemoryInterface
from pyrit.common.path import RESULTS_PATH


class DataExporter:
    """
    Handles the export of data from the database to various formats, currently supporting JSON.
    This class utilizes the strategy design pattern to select the appropriate export format.
    """

    def __init__(
        self, memory_interface: MemoryInterface, *, export_path: Union[Path, str] = None, export_type: str = "json"
    ):
        """Initializes the DataExporter with a memory interface, export path, and export type.

        Args:
            memory_interface (MemoryInterface): The memory interface to interact with the database.
            export_path (Union[Path, str], optional): The path where exported files will be
            stored. Defaults to RESULTS_PATH if not provided
            export_type (str, optional): The format for exporting data. Currently supports 'json'. Defaults to "json".
        """
        self.memory_interface = memory_interface
        self.results_path = Path(export_path) if export_path else RESULTS_PATH
        self.export_type = export_type
        # Using strategy design pattern for export functionality.
        self.export_strategies = {
            "json": self.export_to_json,
            # Future formats can be added here, e.g., "csv": self._export_to_csv
        }

    def export_all_tables(self):
        """
        Exports data for all tables in the database to files, creating one file per
        table in the specified export format.
        """
        table_models = self.memory_interface.get_all_table_models()

        for model in table_models:
            data = self.memory_interface.query_entries(model)
            export_func = self.export_strategies.get(self.export_type)
            if export_func:
                export_func(data, model.__tablename__)

    def export_by_conversation_id(self, conversation_id: str, *, json_suffix: str = "") -> None:
        """
        Exports data associated with a specific conversation ID to a file in the specified export format.
        The filename is constructed using the conversation ID and an optional suffix,
        and it is stored under the results path.

        Args:
            conversation_id (str): The conversation ID for which to export the data.
            json_suffix (str, optional): An optional suffix for the file name. Defaults to an empty string.
        """
        data = self.memory_interface.get_memories_with_conversation_id(conversation_id=conversation_id)

        # Construct the file name using the conversation_id and optional suffix
        filename = (
            f"{conversation_id}{json_suffix}.json" if self.export_type == "json" else f"{conversation_id}{json_suffix}"
        )
        export_func = self.export_strategies.get(self.export_type)
        if export_func:
            export_func(data, filename)

    def export_to_json(self, data: list[Base], table_name: str) -> None:  # type: ignore
        """
        Exports the provided data to a JSON file, naming the file after the table name.
        Each item in the data list, representing a row from the table,
        is converted to a dictionary before being written to the file.

        Args:
            data (list[Base]): The data to be exported, as a list of SQLAlchemy model instances.
            table_name (str): The name of the table, used to construct the file name.
        """
        filename = f"{table_name}.json"
        json_path = self.results_path / filename

        export_data = [self.model_to_dict(instance) for instance in data]
        with open(json_path, "w") as f:
            json.dump(export_data, f, indent=4)

    def model_to_dict(self, model_instance):
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
