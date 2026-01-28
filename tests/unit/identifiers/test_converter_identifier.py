# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Tests for ConverterIdentifier-specific functionality.

Note: Base Identifier functionality (hash computation, to_dict/from_dict basics,
frozen/hashable properties) is tested via ScorerIdentifier in test_scorer_identifier.py.
These tests focus on converter-specific fields and behaviors.
"""

import pytest

from pyrit.identifiers import ConverterIdentifier


class TestConverterIdentifierSpecificFields:
    """Test ConverterIdentifier-specific fields: supported_input_types, supported_output_types, converter_specific_params."""

    def test_supported_input_output_types_stored_as_tuples(self):
        """Test that supported_input_types and supported_output_types are stored correctly."""
        identifier = ConverterIdentifier(
            class_name="TestConverter",
            class_module="pyrit.prompt_converter.test_converter",
            class_description="A test converter",
            identifier_type="instance",
            supported_input_types=("text", "image_path"),
            supported_output_types=("text",),
        )

        assert identifier.supported_input_types == ("text", "image_path")
        assert identifier.supported_output_types == ("text",)

    def test_converter_specific_params_stored(self):
        """Test that converter_specific_params are stored correctly."""
        identifier = ConverterIdentifier(
            class_name="CaesarConverter",
            class_module="pyrit.prompt_converter.caesar_converter",
            class_description="A Caesar cipher converter",
            identifier_type="instance",
            supported_input_types=("text",),
            supported_output_types=("text",),
            converter_specific_params={"shift": 3, "preserve_case": True},
        )

        assert identifier.converter_specific_params["shift"] == 3
        assert identifier.converter_specific_params["preserve_case"] is True

    def test_sub_identifier_with_nested_converter(self):
        """Test that sub_identifier can hold nested ConverterIdentifier."""
        sub_converter = ConverterIdentifier(
            class_name="SubConverter",
            class_module="pyrit.prompt_converter.sub_converter",
            class_description="A sub converter",
            identifier_type="instance",
            supported_input_types=("text",),
            supported_output_types=("text",),
        )

        identifier = ConverterIdentifier(
            class_name="TestConverter",
            class_module="pyrit.prompt_converter.test_converter",
            class_description="A test converter",
            identifier_type="instance",
            supported_input_types=("text",),
            supported_output_types=("text",),
            sub_identifier=[sub_converter],
        )

        assert len(identifier.sub_identifier) == 1
        assert identifier.sub_identifier[0].class_name == "SubConverter"


class TestConverterIdentifierHashDifferences:
    """Test that converter-specific fields affect hash computation."""

    def test_hash_different_for_different_input_types(self):
        """Test that different input types produce different hashes."""
        base_args = {
            "class_name": "TestConverter",
            "class_module": "pyrit.prompt_converter.test_converter",
            "class_description": "A test converter",
            "identifier_type": "instance",
            "supported_output_types": ("text",),
        }

        identifier1 = ConverterIdentifier(supported_input_types=("text",), **base_args)
        identifier2 = ConverterIdentifier(supported_input_types=("image_path",), **base_args)

        assert identifier1.hash != identifier2.hash

    def test_hash_different_for_different_output_types(self):
        """Test that different output types produce different hashes."""
        base_args = {
            "class_name": "TestConverter",
            "class_module": "pyrit.prompt_converter.test_converter",
            "class_description": "A test converter",
            "identifier_type": "instance",
            "supported_input_types": ("text",),
        }

        identifier1 = ConverterIdentifier(supported_output_types=("text",), **base_args)
        identifier2 = ConverterIdentifier(supported_output_types=("image_path",), **base_args)

        assert identifier1.hash != identifier2.hash

    def test_hash_different_for_different_converter_specific_params(self):
        """Test that different converter_specific_params produce different hashes."""
        base_args = {
            "class_name": "CaesarConverter",
            "class_module": "pyrit.prompt_converter.caesar_converter",
            "class_description": "A Caesar cipher converter",
            "identifier_type": "instance",
            "supported_input_types": ("text",),
            "supported_output_types": ("text",),
        }

        identifier1 = ConverterIdentifier(converter_specific_params={"shift": 3}, **base_args)
        identifier2 = ConverterIdentifier(converter_specific_params={"shift": 5}, **base_args)

        assert identifier1.hash != identifier2.hash


class TestConverterIdentifierToDict:
    """Test to_dict includes converter-specific fields."""

    def test_to_dict_includes_supported_types(self):
        """Test that supported_input_types and supported_output_types are in to_dict."""
        identifier = ConverterIdentifier(
            class_name="TestConverter",
            class_module="pyrit.prompt_converter.test_converter",
            class_description="A test converter",
            identifier_type="instance",
            supported_input_types=("text", "image_path"),
            supported_output_types=("text",),
        )

        result = identifier.to_dict()

        # Tuples remain as tuples in to_dict
        assert result["supported_input_types"] == ("text", "image_path")
        assert result["supported_output_types"] == ("text",)

    def test_to_dict_includes_converter_specific_params(self):
        """Test that converter_specific_params are included in to_dict."""
        identifier = ConverterIdentifier(
            class_name="CaesarConverter",
            class_module="pyrit.prompt_converter.caesar_converter",
            class_description="A Caesar cipher converter",
            identifier_type="instance",
            supported_input_types=("text",),
            supported_output_types=("text",),
            converter_specific_params={"shift": 13, "preserve_case": True},
        )

        result = identifier.to_dict()

        assert result["converter_specific_params"] == {"shift": 13, "preserve_case": True}


class TestConverterIdentifierFromDict:
    """Test from_dict handles converter-specific fields."""

    def test_from_dict_converts_lists_to_tuples(self):
        """Test that from_dict converts supported_*_types lists to tuples."""
        data = {
            "class_name": "TestConverter",
            "class_module": "pyrit.prompt_converter.test_converter",
            "class_description": "A test converter",
            "identifier_type": "instance",
            "supported_input_types": ["text", "image_path"],  # List from JSON
            "supported_output_types": ["text"],  # List from JSON
        }

        identifier = ConverterIdentifier.from_dict(data)

        # Lists should be converted to tuples
        assert identifier.supported_input_types == ("text", "image_path")
        assert identifier.supported_output_types == ("text",)
        assert isinstance(identifier.supported_input_types, tuple)
        assert isinstance(identifier.supported_output_types, tuple)

    def test_from_dict_provides_defaults_for_missing_types(self):
        """Test that from_dict provides defaults for missing supported_*_types fields."""
        data = {
            "class_name": "LegacyConverter",
            "class_module": "pyrit.prompt_converter.legacy",
            "class_description": "A legacy converter",
            "identifier_type": "instance",
            # Missing supported_input_types and supported_output_types
        }

        identifier = ConverterIdentifier.from_dict(data)

        # Should provide empty tuples as defaults
        assert identifier.supported_input_types == ()
        assert identifier.supported_output_types == ()

    def test_from_dict_with_nested_sub_identifiers(self):
        """Test from_dict with nested sub_identifier list creates ConverterIdentifier objects."""
        data = {
            "class_name": "PipelineConverter",
            "class_module": "pyrit.prompt_converter.pipeline_converter",
            "class_description": "A pipeline converter",
            "identifier_type": "instance",
            "supported_input_types": ["text"],
            "supported_output_types": ["text"],
            "sub_identifier": [
                {
                    "class_name": "SubConverter",
                    "class_module": "pyrit.prompt_converter.sub",
                    "class_description": "Sub converter",
                    "identifier_type": "instance",
                    "supported_input_types": ["text"],
                    "supported_output_types": ["image_path"],
                },
            ],
        }

        identifier = ConverterIdentifier.from_dict(data)

        assert len(identifier.sub_identifier) == 1
        assert isinstance(identifier.sub_identifier[0], ConverterIdentifier)
        assert identifier.sub_identifier[0].supported_output_types == ("image_path",)

