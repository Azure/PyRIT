# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from dataclasses import dataclass, field
from jinja2 import Template
from pathlib import Path
from typing import Dict, List
import yaml

@dataclass
class ManyShotTemplate:
    template: str
    name: str = ""
    description: str = ""
    should_be_blocked: bool = False
    harm_category: str = ""
    author: str = ""
    group: str = ""
    source: str = ""
    parameters: list[str] = field(default_factory=list)
    
    @classmethod
    def from_yaml_file(cls, file_path: Path):
        """
        Creates an instance of ManyShotTemplate from a YAML file.

        Args:
            file_path (Path): Path to the YAML file.

        Returns:
            ManyShotTemplate: An instance of the class with the YAML content.
        """
        try:
            with open(file_path, "r") as file:
                content = yaml.safe_load(file)  # Safely load YAML content to avoid arbitrary code execution
        except FileNotFoundError:
            raise FileNotFoundError(
                "Invalid dataset file path detected. Please verify that the file is present at the specified "
                f"location: {file_path}."
            )
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing YAML file: {e}")

        if "template" not in content or "parameters" not in content:
            raise ValueError("YAML file must contain 'template' and 'parameters' keys.")

        # Return an instance of the class with loaded parameters
        return cls(template=content["template"], parameters=content["parameters"])

    def apply_parameters(self, prompt: str, examples: List[Dict[str, str]]) -> str:
        # Create a Jinja2 template from the template string
        jinja_template = Template(self.template)

        # Render the template with the provided prompt and examples
        filled_template = jinja_template.render(prompt=prompt, examples=examples)

        return filled_template