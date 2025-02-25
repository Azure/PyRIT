# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import csv
import json
from pathlib import Path

from pyrit.models import PromptRequestPiece


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
        self, data: list[PromptRequestPiece], *, file_path: Path = None, export_type: str = "json"
    ):  # type: ignore
        """
        Exports the provided data to a file in the specified format.

        Args:
            data (list[PromptRequestPiece]): The data to be exported, as a list of PromptRequestPiece instances.
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

    def export_to_json(self, data: list[PromptRequestPiece], file_path: Path = None) -> None:  # type: ignore
        """
        Exports the provided data to a JSON file at the specified file path.
        Each item in the data list, representing a row from the table,
        is converted to a dictionary before being written to the file.

        Args:
            data (list[PromptRequestPiece]): The data to be exported, as a list of PromptRequestPiece instances.
            file_path (Path): The full path, including the file name, where the data will be exported.

        Raises:
            ValueError: If no file_path is provided.
        """
        if not file_path:
            raise ValueError("Please provide a valid file path for exporting data.")
        if not data:
            raise ValueError("No data to export.")
        export_data = []
        for piece in data:
            export_data.append(piece.to_dict())
        with open(file_path, "w") as f:
            json.dump(export_data, f, indent=4)

    def export_to_csv(self, data: list[PromptRequestPiece], file_path: Path = None) -> None:  # type: ignore
        """
        Exports the provided data to a CSV file at the specified file path.
        Each item in the data list, representing a row from the table,
        is converted to a dictionary before being written to the file.

        Args:
            data (list[PromptRequestPiece]): The data to be exported, as a list of PromptRequestPiece instances.
            file_path (Path): The full path, including the file name, where the data will be exported.

        Raises:
            ValueError: If no file_path is provided.
        """
        if not file_path:
            raise ValueError("Please provide a valid file path for exporting data.")
        if not data:
            raise ValueError("No data to export.")
        export_data = []
        for piece in data:
            export_data.append(piece.to_dict())
        fieldnames = list(export_data[0].keys())
        with open(file_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(export_data)
