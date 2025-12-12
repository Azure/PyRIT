# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import pathlib

import pytest

from pyrit.common.path import DATASETS_PATH
from pyrit.datasets import TextJailBreak


@pytest.fixture
def jailbreak_dir():
    return pathlib.Path(DATASETS_PATH) / "jailbreak" / "templates"


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


def test_all_templates_render_without_syntax_errors(jailbreak_dir):
    """Test that all jailbreak templates can be successfully rendered with a test prompt."""
    yaml_files = [f for f in jailbreak_dir.rglob("*.yaml") if "multi_parameter" not in f.parts]
    assert len(yaml_files) > 0, "No YAML templates found in jailbreak directory"

    test_prompt = "This is a test prompt for template validation"
    failed_templates = []

    for template_file in yaml_files:
        try:
            jailbreak = TextJailBreak(template_path=str(template_file))
            # Attempt to render the template - this will catch Jinja2 syntax errors
            result = jailbreak.get_jailbreak(test_prompt)
            # Verify the prompt was inserted and template placeholders were removed
            assert test_prompt in result, f"Template {template_file.name} did not insert test prompt"
            assert "{{ prompt }}" not in result, f"Template {template_file.name} still contains template placeholder"
        except Exception as e:
            failed_templates.append((template_file.relative_to(jailbreak_dir), str(e)))

    if failed_templates:
        error_msg = "The following templates failed to render:\n"
        for template_path, error in failed_templates:
            error_msg += f"  - {template_path}: {error}\n"
        pytest.fail(error_msg)


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


def test_random_template_validation_fails_on_invalid_syntax():
    """Test that random template selection validates syntax and fails with clear error on invalid templates."""
    # Create a temporary invalid template to test validation
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        invalid_template_path = tmpdir_path / "invalid.yaml"

        # Create a template with invalid Jinja2 syntax
        invalid_template_path.write_text(
            """---
name: test_invalid_template
parameters:
  - prompt
data_type: text
value: |
  This has invalid syntax: {{ '{' }}{{ prompt }}{{ '}' }}
"""
        )

        # Try to load and validate it
        with pytest.raises(ValueError) as exc_info:
            jailbreak = TextJailBreak(template_path=str(invalid_template_path))
            jailbreak.get_jailbreak("test")

        # Verify error message contains helpful context from seed.py
        error_msg = str(exc_info.value)
        assert "Error rendering template" in error_msg


def test_template_source_tracking(jailbreak_dir):
    """Test that template source is tracked for better error reporting."""
    template_path = jailbreak_dir / "dan_1.yaml"
    jailbreak = TextJailBreak(template_path=str(template_path))

    # Verify template_source is set
    assert hasattr(jailbreak, "template_source")
    assert "dan_1.yaml" in str(jailbreak.template_source)

    # Test with string template
    jailbreak_string = TextJailBreak(string_template="Test {{ prompt }}")
    assert jailbreak_string.template_source == "<string_template>"
