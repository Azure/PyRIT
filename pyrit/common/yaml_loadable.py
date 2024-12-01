# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import abc
import yaml
from pathlib import Path
from typing import Type, TypeVar, Union


T = TypeVar("T", bound="YamlLoadable")


class YamlLoadable(abc.ABC):
    """
    Abstract base class for objects that can be loaded from YAML files.
    """

    @classmethod
    def from_yaml_file(cls: Type[T], file: Union[Path | str]) -> T:
        """
        Creates a new object from a YAML file.

        Args:
            file: The input file path.

        Returns:
            A new object of type T.

        Raises:
            FileNotFoundError: If the input YAML file path does not exist.
            ValueError: If the YAML file is invalid.
        """
        file = Path(file)
        if not file.exists():
            raise FileNotFoundError(f"File '{file}' does not exist.")
        try:
            yaml_data = yaml.safe_load(file.read_text("utf-8"))
        except yaml.YAMLError as exc:
            raise ValueError(f"Invalid YAML file '{file}': {exc}")
        data_object = cls(**yaml_data)
        return data_object
