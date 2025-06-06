# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import pathlib
import random

from pyrit.common.path import DATASETS_PATH
from pyrit.models import SeedPrompt


class TextJailBreak:
    """
    A class that manages jailbreak datasets (like DAN, etc.)
    """

    def __init__(
        self,
        *,
        template_path=None,
        template_file_name=None,
        string_template=None,
        random_template=False,
    ):
        """
        Initialize a Jailbreak instance with exactly one template source.

        Args:
            template_path (str, optional): Full path to a YAML template file.
            template_file_name (str, optional): Name of a template file in datasets/jailbreak directory.
            string_template (str, optional): A string template to use directly.
            random_template (bool, optional): Whether to use a random template from datasets/jailbreak.

        Raises:
            ValueError: If more than one template source is provided or if no template source is provided.
        """
        # Count how many template sources are provided
        template_sources = [template_path, template_file_name, string_template, random_template]
        provided_sources = [source for source in template_sources if source]

        if len(provided_sources) != 1:
            raise ValueError(
                "Exactly one of template_path, template_file_name, string_template, or random_template must be provided"
            )

        if template_path:
            self.template = SeedPrompt.from_yaml_file(template_path)
        elif string_template:
            self.template = SeedPrompt(value=string_template)
        else:
            # Get all yaml files in the jailbreak directory and its subdirectories
            jailbreak_dir = pathlib.Path(DATASETS_PATH) / "jailbreak"
            yaml_files = list(jailbreak_dir.rglob("*.yaml"))
            if not yaml_files:
                raise ValueError("No YAML templates found in jailbreak directory or its subdirectories")

            if template_file_name:
                matching_files = [f for f in yaml_files if f.name == template_file_name]
                if not matching_files:
                    raise ValueError(
                        f"Template file '{template_file_name}' not found in jailbreak directory or its subdirectories"
                    )
                if len(matching_files) > 1:
                    raise ValueError(f"Multiple files named '{template_file_name}' found in jailbreak directory")
                self.template = SeedPrompt.from_yaml_file(matching_files[0])
            else:
                random_template_path = random.choice(yaml_files)
                self.template = SeedPrompt.from_yaml_file(random_template_path)

    def get_jailbreak_system_prompt(self):
        return self.get_jailbreak(prompt="")

    def get_jailbreak(self, prompt: str):
        return self.template.render_template_value(prompt=prompt)
