# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


import pytest

from pyrit.models import Message, MessagePiece
from pyrit.prompt_target import PromptTarget
from pyrit.registry.instance_registries.target_registry import TargetRegistry


class MockPromptTarget(PromptTarget):
    """Mock PromptTarget for testing."""

    def __init__(self, *, model_name: str = "mock_model") -> None:
        super().__init__(model_name=model_name)

    async def send_prompt_async(
        self,
        *,
        message: Message,
    ) -> list[Message]:
        return [
            MessagePiece(
                role="assistant",
                original_value="mock response",
            ).to_message()
        ]

    def _validate_request(self, *, message: Message) -> None:
        pass

    async def dispose_async(self) -> None:
        pass


class MockChatTarget(PromptTarget):
    """Mock chat target for testing different target types."""

    def __init__(self, *, endpoint: str = "http://test") -> None:
        super().__init__(endpoint=endpoint)

    async def send_prompt_async(
        self,
        *,
        message: Message,
    ) -> list[Message]:
        return [
            MessagePiece(
                role="assistant",
                original_value="chat response",
            ).to_message()
        ]

    def _validate_request(self, *, message: Message) -> None:
        pass

    async def dispose_async(self) -> None:
        pass


class TestTargetRegistrySingleton:
    """Tests for the singleton pattern in TargetRegistry."""

    def setup_method(self):
        """Reset the singleton before each test."""
        TargetRegistry.reset_instance()

    def teardown_method(self):
        """Reset the singleton after each test."""
        TargetRegistry.reset_instance()

    def test_get_registry_singleton_returns_same_instance(self):
        """Test that get_registry_singleton returns the same singleton each time."""
        instance1 = TargetRegistry.get_registry_singleton()
        instance2 = TargetRegistry.get_registry_singleton()

        assert instance1 is instance2

    def test_get_registry_singleton_returns_target_registry_type(self):
        """Test that get_registry_singleton returns a TargetRegistry instance."""
        instance = TargetRegistry.get_registry_singleton()
        assert isinstance(instance, TargetRegistry)

    def test_reset_instance_clears_singleton(self):
        """Test that reset_instance clears the singleton."""
        instance1 = TargetRegistry.get_registry_singleton()
        TargetRegistry.reset_instance()
        instance2 = TargetRegistry.get_registry_singleton()

        assert instance1 is not instance2


@pytest.mark.usefixtures("patch_central_database")
class TestTargetRegistryRegisterInstance:
    """Tests for register_instance functionality in TargetRegistry."""

    def setup_method(self):
        """Reset and get a fresh registry for each test."""
        TargetRegistry.reset_instance()
        self.registry = TargetRegistry.get_registry_singleton()

    def teardown_method(self):
        """Reset the singleton after each test."""
        TargetRegistry.reset_instance()

    def test_register_instance_with_custom_name(self):
        """Test registering a target with a custom name."""
        target = MockPromptTarget()
        self.registry.register_instance(target, name="custom_target")

        assert "custom_target" in self.registry
        assert self.registry.get("custom_target") is target

    def test_register_instance_generates_name_from_class(self):
        """Test that register_instance generates a name from class name when not provided."""
        target = MockPromptTarget()
        self.registry.register_instance(target)

        # Name should be derived from class name with hash suffix
        names = self.registry.get_names()
        assert len(names) == 1
        assert names[0].startswith("mock_prompt_")

    def test_register_instance_multiple_targets_unique_names(self):
        """Test registering multiple targets generates unique names."""
        target1 = MockPromptTarget()
        target2 = MockChatTarget()

        self.registry.register_instance(target1)
        self.registry.register_instance(target2)

        assert len(self.registry) == 2

    def test_register_instance_same_target_type_different_config(self):
        """Test that same target class with different configs can be registered."""
        target1 = MockPromptTarget(model_name="model_a")
        target2 = MockPromptTarget(model_name="model_b")

        # Register with explicit names
        self.registry.register_instance(target1, name="target_1")
        self.registry.register_instance(target2, name="target_2")

        assert len(self.registry) == 2


@pytest.mark.usefixtures("patch_central_database")
class TestTargetRegistryGetInstanceByName:
    """Tests for get_instance_by_name functionality in TargetRegistry."""

    def setup_method(self):
        """Reset and get a fresh registry for each test."""
        TargetRegistry.reset_instance()
        self.registry = TargetRegistry.get_registry_singleton()
        self.target = MockPromptTarget()
        self.registry.register_instance(self.target, name="test_target")

    def teardown_method(self):
        """Reset the singleton after each test."""
        TargetRegistry.reset_instance()

    def test_get_instance_by_name_returns_target(self):
        """Test getting a registered target by name."""
        result = self.registry.get_instance_by_name("test_target")
        assert result is self.target

    def test_get_instance_by_name_nonexistent_returns_none(self):
        """Test that getting a non-existent target returns None."""
        result = self.registry.get_instance_by_name("nonexistent")
        assert result is None


@pytest.mark.usefixtures("patch_central_database")
class TestTargetRegistryBuildMetadata:
    """Tests for _build_metadata functionality in TargetRegistry."""

    def setup_method(self):
        """Reset and get a fresh registry for each test."""
        TargetRegistry.reset_instance()
        self.registry = TargetRegistry.get_registry_singleton()

    def teardown_method(self):
        """Reset the singleton after each test."""
        TargetRegistry.reset_instance()

    def test_build_metadata_includes_class_name(self):
        """Test that metadata includes the class name."""
        target = MockPromptTarget()
        self.registry.register_instance(target, name="mock_target")

        metadata = self.registry.list_metadata()
        assert len(metadata) == 1
        assert metadata[0].class_name == "MockPromptTarget"
        assert metadata[0].name == "mock_target"

    def test_build_metadata_includes_target_identifier(self):
        """Test that metadata includes the target_identifier."""
        target = MockPromptTarget(model_name="test_model")
        self.registry.register_instance(target, name="mock_target")

        metadata = self.registry.list_metadata()
        assert hasattr(metadata[0], "target_identifier")
        assert isinstance(metadata[0].target_identifier, dict)
        assert metadata[0].target_identifier.get("model_name") == "test_model"

    def test_build_metadata_description_from_docstring(self):
        """Test that description is derived from the target's docstring."""
        target = MockPromptTarget()
        self.registry.register_instance(target, name="mock_target")

        metadata = self.registry.list_metadata()
        # MockPromptTarget has a docstring
        assert "Mock PromptTarget for testing" in metadata[0].description


@pytest.mark.usefixtures("patch_central_database")
class TestTargetRegistryListMetadata:
    """Tests for list_metadata in TargetRegistry."""

    def setup_method(self):
        """Reset and get a fresh registry with multiple targets."""
        TargetRegistry.reset_instance()
        self.registry = TargetRegistry.get_registry_singleton()

        self.target1 = MockPromptTarget(model_name="model_a")
        self.target2 = MockPromptTarget(model_name="model_b")
        self.chat_target = MockChatTarget()

        self.registry.register_instance(self.target1, name="target_1")
        self.registry.register_instance(self.target2, name="target_2")
        self.registry.register_instance(self.chat_target, name="chat_target")

    def teardown_method(self):
        """Reset the singleton after each test."""
        TargetRegistry.reset_instance()

    def test_list_metadata_returns_all_registered(self):
        """Test that list_metadata returns metadata for all registered targets."""
        metadata = self.registry.list_metadata()
        assert len(metadata) == 3

    def test_list_metadata_filter_by_class_name(self):
        """Test filtering metadata by class_name."""
        mock_metadata = self.registry.list_metadata(include_filters={"class_name": "MockPromptTarget"})

        assert len(mock_metadata) == 2
        for m in mock_metadata:
            assert m.class_name == "MockPromptTarget"


@pytest.mark.usefixtures("patch_central_database")
class TestTargetRegistryComputeIdentifierHash:
    """Tests for _compute_identifier_hash functionality."""

    def setup_method(self):
        """Reset the singleton before each test."""
        TargetRegistry.reset_instance()

    def teardown_method(self):
        """Reset the singleton after each test."""
        TargetRegistry.reset_instance()

    def test_compute_identifier_hash_deterministic(self):
        """Test that identifier hash is deterministic for same config."""
        target1 = MockPromptTarget(model_name="same_model")
        target2 = MockPromptTarget(model_name="same_model")

        hash1 = TargetRegistry._compute_identifier_hash(target1)
        hash2 = TargetRegistry._compute_identifier_hash(target2)

        assert hash1 == hash2

    def test_compute_identifier_hash_different_for_different_config(self):
        """Test that identifier hash is different for different configs."""
        target1 = MockPromptTarget(model_name="model_a")
        target2 = MockPromptTarget(model_name="model_b")

        hash1 = TargetRegistry._compute_identifier_hash(target1)
        hash2 = TargetRegistry._compute_identifier_hash(target2)

        assert hash1 != hash2

    def test_compute_identifier_hash_is_string(self):
        """Test that identifier hash returns a hex string."""
        target = MockPromptTarget()
        hash_value = TargetRegistry._compute_identifier_hash(target)

        assert isinstance(hash_value, str)
        # Should be a valid hex string (SHA256 = 64 hex chars)
        assert len(hash_value) == 64
        assert all(c in "0123456789abcdef" for c in hash_value)
