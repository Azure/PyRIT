# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import re
import yaml
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Union
from jinja2 import Template

from pyrit.common.yaml_loadable import YamlLoadable


@dataclass
class PromptTemplate(YamlLoadable):
    template: str
    name: str = ""
    description: str = ""
    should_be_blocked: bool = False
    harm_category: str = ""
    author: str = ""
    group: str = ""
    source: str = ""
    parameters: list[str] = field(default_factory=list)

    def apply_custom_metaprompt_parameters(self, **kwargs) -> str:
        """Builds a new prompts from the metaprompt template.
        Args:
            **kwargs: the key value for the metaprompt template inputs

        Returns:
            A new prompt following the template
        """
        final_prompt = self.template
        for key, value in kwargs.items():
            if key not in self.parameters:
                raise ValueError(f'Invalid parameters passed. [expected="{self.parameters}", actual="{kwargs}"]')
            # Matches field names within brackets {{ }}
            #  {{   key    }}
            #  ^^^^^^^^^^^^^^
            regex = "{}{}{}".format("\{\{ *", key, " *\}\}")  # noqa: W605
            matches = re.findall(pattern=regex, string=final_prompt)
            if not matches:
                raise ValueError(
                    f"No parameters matched, they might be missing in the template. "
                    f'[expected="{self.parameters}", actual="{kwargs}"]'
                )
            final_prompt = re.sub(pattern=regex, string=final_prompt, repl=str(value))
        return final_prompt


@dataclass
class AttackStrategy:
    def __init__(self, *, strategy: Union[Path | str], **kwargs):
        self.kwargs = kwargs
        if isinstance(strategy, Path):
            self.strategy = PromptTemplate.from_yaml_file(strategy)
        else:
            self.strategy = PromptTemplate(template=strategy, parameters=list(kwargs.keys()))

    def __str__(self):
        """Returns a string representation of the attack strategy."""
        return self.strategy.apply_custom_metaprompt_parameters(**self.kwargs)


@dataclass
class ManyShotTemplate(PromptTemplate):
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
