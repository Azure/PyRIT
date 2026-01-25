# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Tests for backend registry service.
"""



from pyrit.backend.services.registry_service import (
    RegistryService,
    _extract_params_schema,
    _get_all_subclasses,
    get_registry_service,
)


class TestExtractParamsSchema:
    """Tests for _extract_params_schema helper function."""

    def test_extract_params_with_required_and_optional(self) -> None:
        """Test extracting params from a class with required and optional params."""

        class TestClass:
            def __init__(self, required_param: str, optional_param: str = "default") -> None:
                pass

        result = _extract_params_schema(TestClass)

        assert "required_param" in result["required"]
        assert "optional_param" in result["optional"]

    def test_extract_params_ignores_self(self) -> None:
        """Test that self is ignored in param extraction."""

        class TestClass:
            def __init__(self, param: str) -> None:
                pass

        result = _extract_params_schema(TestClass)

        assert "self" not in result["required"]
        assert "self" not in result["optional"]


class TestGetAllSubclasses:
    """Tests for _get_all_subclasses helper function."""

    def test_get_subclasses_finds_concrete_classes(self) -> None:
        """Test that concrete subclasses are found."""

        class Base:
            pass

        class Child1(Base):
            pass

        class Child2(Base):
            pass

        result = _get_all_subclasses(Base)

        assert Child1 in result
        assert Child2 in result


class TestRegistryService:
    """Tests for RegistryService."""

    def test_get_targets_returns_list(self) -> None:
        """Test that get_targets returns a list."""
        service = RegistryService()

        result = service.get_targets()

        assert isinstance(result, list)

    def test_get_targets_filters_chat_targets(self) -> None:
        """Test that get_targets can filter by chat target support."""
        service = RegistryService()

        chat_only = service.get_targets(is_chat_target=True)
        non_chat = service.get_targets(is_chat_target=False)

        # Chat targets and non-chat targets should be different
        chat_names = {t.name for t in chat_only}
        non_chat_names = {t.name for t in non_chat}
        # They should be disjoint (no overlap)
        assert len(chat_names & non_chat_names) == 0

    def test_get_converters_returns_list(self) -> None:
        """Test that get_converters returns a list."""
        service = RegistryService()

        result = service.get_converters()

        assert isinstance(result, list)

    def test_get_converters_filters_llm_based(self) -> None:
        """Test that get_converters can filter by LLM-based status."""
        service = RegistryService()

        llm_based = service.get_converters(is_llm_based=True)
        non_llm = service.get_converters(is_llm_based=False)

        # LLM-based and non-LLM converters should be different
        llm_names = {c.name for c in llm_based}
        non_llm_names = {c.name for c in non_llm}
        # They should be disjoint
        assert len(llm_names & non_llm_names) == 0

    def test_get_scenarios_returns_list(self) -> None:
        """Test that get_scenarios returns a list."""
        service = RegistryService()

        result = service.get_scenarios()

        assert isinstance(result, list)

    def test_get_scorers_returns_list(self) -> None:
        """Test that get_scorers returns a list."""
        service = RegistryService()

        result = service.get_scorers()

        assert isinstance(result, list)

    def test_get_initializers_returns_list(self) -> None:
        """Test that get_initializers returns a list."""
        service = RegistryService()

        result = service.get_initializers()

        assert isinstance(result, list)


class TestGetRegistryServiceSingleton:
    """Tests for get_registry_service singleton function."""

    def test_returns_same_instance(self) -> None:
        """Test that get_registry_service returns the same instance."""
        # Reset singleton for test
        import pyrit.backend.services.registry_service as module

        module._registry_service = None

        service1 = get_registry_service()
        service2 = get_registry_service()

        assert service1 is service2

    def test_returns_registry_service_instance(self) -> None:
        """Test that get_registry_service returns a RegistryService."""
        service = get_registry_service()

        assert isinstance(service, RegistryService)
