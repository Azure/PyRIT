# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import pytest

from pyrit.common.path import JAILBREAK_TEMPLATES_PATH
from pyrit.datasets import TextJailBreak


@pytest.fixture
def jailbreak_dir():
    return JAILBREAK_TEMPLATES_PATH


def test_init_with_template_path(jailbreak_dir):
    template_path = jailbreak_dir / "dan_1.yaml"
    jailbreak = TextJailBreak(template_path=str(template_path))
    assert "DAN" in jailbreak.template.value
    assert "{{ prompt }}" in jailbreak.template.value


def test_init_with_template_file_name(jailbreak_dir):
    jailbreak = TextJailBreak(template_file_name="dan_1.yaml")
    assert "DAN" in jailbreak.template.value
    assert "{{ prompt }}" in jailbreak.template.value


def test_init_with_string_template():
    template = "Custom template {{ prompt }}"
    jailbreak = TextJailBreak(string_template=template)
    assert jailbreak.template.value == template


def test_init_with_random_template(jailbreak_dir):
    jailbreak = TextJailBreak(random_template=True)
    assert jailbreak.template.value is not None
    assert "{{ prompt }}" in jailbreak.template.value


def test_init_no_template_source():
    with pytest.raises(ValueError, match="Exactly one of template_path"):
        TextJailBreak()


def test_init_multiple_template_sources():
    with pytest.raises(ValueError, match="Exactly one of template_path"):
        TextJailBreak(template_path="path", template_file_name="name")


def test_init_template_file_not_found():
    with pytest.raises(ValueError, match="Template file 'nonexistent.yaml' not found"):
        TextJailBreak(template_file_name="nonexistent.yaml")


def test_get_jailbreak_as_system_prompt(jailbreak_dir):
    template_path = jailbreak_dir / "dan_1.yaml"
    jailbreak = TextJailBreak(template_path=str(template_path))
    system_prompt = jailbreak.get_jailbreak_system_prompt()
    assert "DAN" in system_prompt
    assert "{{ prompt }}" not in system_prompt


def test_get_jailbreak_with_prompt_inserted(jailbreak_dir):
    template_path = jailbreak_dir / "dan_1.yaml"
    jailbreak = TextJailBreak(template_path=str(template_path))
    result = jailbreak.get_jailbreak("Tell me a joke")
    assert "DAN" in result
    assert "Tell me a joke" in result
    assert "{{ prompt }}" not in result


def test_get_file_name_subdirectory():
    filename = "nova.yaml"
    jailbreak = TextJailBreak(template_file_name=filename)
    result = jailbreak.get_jailbreak("Tell me a joke")
    assert "Tell me a joke" in result
    assert "{{ prompt }}" not in result


def test_all_templates_have_single_prompt_parameter(jailbreak_dir):
    """Test that all jailbreak template files have exactly one prompt parameter in their definition."""
    yaml_files = [f for f in jailbreak_dir.rglob("*.yaml") if "multi_parameter" not in f.parts]
    assert len(yaml_files) > 0, "No YAML templates found in jailbreak directory"

    for template_file in yaml_files:
        try:
            jailbreak = TextJailBreak(template_path=str(template_file))
            template_params = jailbreak.template.parameters

            prompt_params = [p for p in template_params if p == "prompt"]
            assert (
                len(prompt_params) == 1
            ), f"Template {template_file.name} has {len(prompt_params)} prompt parameters in definition, expected 1"
            assert (
                len(template_params) == 1
            ), f"Template {template_file.name} has {len(template_params)} parameters in definition, expected 1"
            if len(template_params) > 1:
                print(f"Template {template_file.name} has {len(template_params)} parameters in definition, expected 1")
        except Exception as e:
            pytest.fail(f"Failed processing template file '{template_file.name}': {str(e)}")


def test_template_with_multiple_parameters_success(jailbreak_dir):
    """Test that a template with multiple parameters works when all required parameters are provided."""
    template_path = jailbreak_dir / "multi_parameter" / "dt_stereotypes_targeted.yaml"
    jailbreak = TextJailBreak(template_path=str(template_path), target_group="specific_group")
    result = jailbreak.get_jailbreak("Tell me a joke")
    assert "specific_group" in result
    assert "Tell me a joke" in result
    assert "{{ prompt }}" not in result
    assert "{{ target_group }}" not in result


def test_template_with_multiple_parameters_missing(jailbreak_dir):
    """Test that an error is raised when required parameters are missing."""
    template_path = jailbreak_dir / "multi_parameter" / "dt_stereotypes_targeted.yaml"

    with pytest.raises(ValueError, match="Template requires parameters that were not provided"):
        TextJailBreak(
            template_path=str(template_path),
        )


def test_template_with_multiple_parameters_prompt_ignored(jailbreak_dir):
    """Test that providing 'prompt' in kwargs is ignored since it's handled separately."""
    template_path = jailbreak_dir / "multi_parameter" / "dt_stereotypes_targeted.yaml"
    jailbreak = TextJailBreak(
        template_path=str(template_path),
        target_group="specific_group",
        prompt="This should be ignored",  # This should be ignored in init
    )
    result = jailbreak.get_jailbreak("Tell me a joke")
    assert "specific_group" in result
    assert "Tell me a joke" in result  # This should be the actual prompt used
    assert "This should be ignored" not in result  # The prompt from kwargs should not appear
    assert "{{ prompt }}" not in result
    assert "{{ target_group }}" not in result
