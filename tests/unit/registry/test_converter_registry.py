# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


import pytest

from pyrit.identifiers import ConverterIdentifier
from pyrit.registry.instance_registries.converter_registry import ConverterRegistry


class MockTextConverter:
    """Mock converter with text input/output support."""

    SUPPORTED_INPUT_TYPES = ("text",)
    SUPPORTED_OUTPUT_TYPES = ("text",)


class MockImageConverter:
    """Mock converter with image input and text output support."""

    SUPPORTED_INPUT_TYPES = ("image_path",)
    SUPPORTED_OUTPUT_TYPES = ("text",)


class TestConverterRegistrySingleton:
    """Tests for singleton behavior in ConverterRegistry."""

    def setup_method(self):
        ConverterRegistry.reset_instance()

    def teardown_method(self):
        ConverterRegistry.reset_instance()

    def test_get_registry_singleton_returns_same_instance(self):
        instance1 = ConverterRegistry.get_registry_singleton()
        instance2 = ConverterRegistry.get_registry_singleton()

        assert instance1 is instance2

    def test_get_registry_singleton_returns_converter_registry_type(self):
        instance = ConverterRegistry.get_registry_singleton()

        assert isinstance(instance, ConverterRegistry)


@pytest.mark.usefixtures("patch_central_database")
class TestConverterRegistryRegistrationAndRetrieval:
    """Tests for registration and lookup behavior."""

    def setup_method(self):
        ConverterRegistry.reset_instance()
        self.registry = ConverterRegistry.get_registry_singleton()

    def teardown_method(self):
        ConverterRegistry.reset_instance()

    def test_register_instance_with_custom_name(self):
        converter = MockTextConverter()

        self.registry.register_instance(converter, name="custom_converter")

        assert self.registry.get_instance_by_name("custom_converter") is converter

    def test_register_instance_generates_name_from_class(self):
        converter = MockTextConverter()

        self.registry.register_instance(converter)

        assert self.registry.get_names() == ["mock_text"]

    def test_get_instance_by_name_nonexistent_returns_none(self):
        assert self.registry.get_instance_by_name("does_not_exist") is None


@pytest.mark.usefixtures("patch_central_database")
class TestConverterRegistryMetadata:
    """Tests metadata generation for registered converters."""

    def setup_method(self):
        ConverterRegistry.reset_instance()
        self.registry = ConverterRegistry.get_registry_singleton()

    def teardown_method(self):
        ConverterRegistry.reset_instance()

    def test_list_metadata_returns_converter_identifier_with_expected_fields(self):
        converter = MockImageConverter()
        self.registry.register_instance(converter, name="image_converter")

        metadata = self.registry.list_metadata()

        assert len(metadata) == 1
        assert isinstance(metadata[0], ConverterIdentifier)
        assert metadata[0].class_name == "MockImageConverter"
        assert metadata[0].class_description == "Converter: image_converter"
        assert metadata[0].supported_input_types == ("image_path",)
        assert metadata[0].supported_output_types == ("text",)
