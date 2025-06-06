# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import pathlib

import pytest

from pyrit.common.path import DATASETS_PATH
from pyrit.datasets.text_jailbreak import TextJailBreak


@pytest.fixture
def jailbreak_dir():
    return pathlib.Path(DATASETS_PATH) / "jailbreak"


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
