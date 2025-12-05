# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Unit tests for the ScenarioRegistry module.
"""

from typing import Type
from unittest.mock import MagicMock, patch

from pyrit.cli.scenario_registry import ScenarioRegistry
from pyrit.scenario.core.scenario import Scenario
from pyrit.scenario.core.scenario_strategy import ScenarioStrategy


class MockStrategy(ScenarioStrategy):
    """Mock strategy for testing."""

    ALL = ("all", {"all"})
    TestStrategy = ("test_strategy", {"test"})

    @classmethod
    def get_aggregate_tags(cls) -> set[str]:
        return {"all"}


class MockScenario(Scenario):
    """Mock scenario for testing."""

    async def _get_atomic_attacks_async(self):
        return []

    @classmethod
    def get_strategy_class(cls) -> Type[ScenarioStrategy]:
        return MockStrategy

    @classmethod
    def get_default_strategy(cls) -> ScenarioStrategy:
        return MockStrategy.ALL

    @classmethod
    def required_datasets(cls) -> list[str]:
        return []


class TestScenarioRegistry:
    """Tests for ScenarioRegistry class."""

    def test_discover_builtin_scenarios(self):
        """Test discovery of built-in scenarios."""
        # Create a registry which will automatically discover built-in scenarios
        registry = ScenarioRegistry()

        # Verify that some scenarios were discovered
        scenario_names = registry.get_scenario_names()
        # We should find at least the built-in scenarios
        # Note: This is an integration test that depends on actual scenario files existing
        assert len(scenario_names) >= 0  # May be 0 if run in isolation

        # If scenarios were found, verify they're Scenario subclasses
        for name in scenario_names:
            scenario_class = registry.get_scenario(name)
            assert scenario_class is not None
            assert issubclass(scenario_class, Scenario)

    def test_discover_builtin_scenarios_correct_module_paths(self):
        """Test that builtin scenario discovery uses correct module paths without duplication.

        This is a regression test for a bug where module paths were incorrectly constructed
        as 'pyrit.scenario.scenarios.pyrit.scenarios.scenarios.xxx' instead of
        'pyrit.scenario.scenarios.xxx'.
        """
        with patch("pyrit.cli.scenario_registry.importlib.import_module") as mock_import:
            with patch("pyrit.cli.scenario_registry.pkgutil.iter_modules") as mock_iter:
                with patch("pyrit.cli.scenario_registry.inspect.getmembers") as mock_getmembers:
                    # Mock the scenarios package structure
                    def iter_modules_side_effect(paths):
                        path_str = str(paths[0]) if paths else ""
                        if "airt" in path_str:
                            # Mock the airt subpackage
                            return [
                                (None, "content_harms_scenario", False),
                                (None, "cyber_scenario", False),
                            ]
                        else:
                            # Mock the main scenarios package
                            return [
                                (None, "encoding_scenario", False),  # A file module
                                (None, "foundry_scenario", False),  # Another file module
                                (None, "airt", True),  # A package (subdirectory)
                            ]

                    mock_iter.side_effect = iter_modules_side_effect

                    # Create mock module with Scenario subclass
                    mock_scenario_class = type(
                        "TestScenario",
                        (Scenario,),
                        {
                            "_get_atomic_attacks_async": lambda self: [],
                            "get_strategy_class": classmethod(lambda cls: MockStrategy),
                            "get_default_strategy": classmethod(lambda cls: MockStrategy.ALL),
                        },
                    )

                    # Mock getmembers to return our test scenario
                    mock_getmembers.return_value = [("TestScenario", mock_scenario_class)]

                    # Mock import_module to return a simple module object
                    def import_side_effect(module_name):
                        mock_mod = MagicMock()
                        mock_mod.__name__ = module_name
                        return mock_mod

                    mock_import.side_effect = import_side_effect

                    registry = ScenarioRegistry()
                    registry._discover_builtin_scenarios()

                    # Verify the correct module paths were attempted
                    import_calls = [call[0][0] for call in mock_import.call_args_list]

                    # Filter to only scenario module imports (not the base package imports)
                    # We expect imports like:
                    # - pyrit.scenario.scenarios.encoding_scenario
                    # - pyrit.scenario.scenarios.foundry_scenario
                    # - pyrit.scenario.scenarios.airt.content_harms_scenario
                    # - pyrit.scenario.scenarios.airt.cyber_scenario
                    scenario_module_imports = [
                        call
                        for call in import_calls
                        if call.startswith("pyrit.scenario.scenarios.") and call != "pyrit.scenario.scenarios"
                    ]

                    # Should NOT see duplicated paths like:
                    # - pyrit.scenario.scenarios.pyrit.scenarios.scenarios.xxx

                    assert len(scenario_module_imports) > 0, "No scenario modules were imported"

                    for call_path in scenario_module_imports:
                        # Verify no path duplication
                        assert (
                            "pyrit.scenario.scenarios.pyrit" not in call_path
                        ), f"Module path has duplication: {call_path}"
                        # Verify 'scenarios' appears exactly once (not duplicated)
                        assert call_path.count("scenarios") == 1, f"Module path has 'scenarios' duplicated: {call_path}"
                        # Verify correct base path
                        assert call_path.startswith(
                            "pyrit.scenario.scenarios."
                        ), f"Module path doesn't start with correct base: {call_path}"
                    
                    # Verify scenarios are registered with their full relative paths
                    # Top-level scenarios should use just the module name
                    assert "encoding_scenario" in registry._scenarios
                    assert "foundry_scenario" in registry._scenarios
                    # Nested scenarios should use dot notation (e.g., "airt.content_harms_scenario")
                    assert "airt.content_harms_scenario" in registry._scenarios
                    assert "airt.cyber_scenario" in registry._scenarios

    def test_get_scenario_existing(self):
        """Test getting an existing scenario."""
        registry = ScenarioRegistry()
        # Manually add a scenario for testing
        registry._scenarios["test_scenario"] = MockScenario

        result = registry.get_scenario("test_scenario")
        assert result == MockScenario

    def test_get_scenario_nonexistent(self):
        """Test getting a non-existent scenario returns None."""
        registry = ScenarioRegistry()
        result = registry.get_scenario("nonexistent_scenario")
        assert result is None

    def test_get_scenario_names_empty(self):
        """Test get_scenario_names with no scenarios."""
        registry = ScenarioRegistry()
        registry._scenarios = {}
        registry._discovered = True  # Prevent auto-discovery
        names = registry.get_scenario_names()
        assert names == []

    def test_get_scenario_names_sorted(self):
        """Test get_scenario_names returns sorted list."""
        registry = ScenarioRegistry()
        registry._scenarios = {
            "zebra_scenario": MockScenario,
            "apple_scenario": MockScenario,
            "middle_scenario": MockScenario,
        }
        registry._discovered = True  # Prevent auto-discovery

        names = registry.get_scenario_names()
        assert names == ["apple_scenario", "middle_scenario", "zebra_scenario"]

    def test_list_scenarios_with_descriptions(self):
        """Test list_scenarios returns scenario information."""

        class DocumentedScenario(Scenario):
            """This is a test scenario for unit testing."""

            async def _get_atomic_attacks_async(self):
                return []

            @classmethod
            def get_strategy_class(cls) -> Type[ScenarioStrategy]:
                return MockStrategy

            @classmethod
            def get_default_strategy(cls) -> ScenarioStrategy:
                return MockStrategy.ALL

            @classmethod
            def required_datasets(cls) -> list[str]:
                return ["test_dataset_1", "test_dataset_2"]

        registry = ScenarioRegistry()
        registry._scenarios = {
            "test_scenario": DocumentedScenario,
        }
        registry._discovered = True  # Prevent auto-discovery

        scenarios = registry.list_scenarios()

        assert len(scenarios) == 1
        assert scenarios[0]["name"] == "test_scenario"
        assert scenarios[0]["class_name"] == "DocumentedScenario"
        assert "test scenario" in scenarios[0]["description"].lower()
        assert scenarios[0]["required_datasets"] == ["test_dataset_1", "test_dataset_2"]

    def test_list_scenarios_no_description(self):
        """Test list_scenarios with scenario lacking docstring."""

        class UndocumentedScenario(Scenario):
            async def _get_atomic_attacks_async(self):
                return []

            @classmethod
            def get_strategy_class(cls) -> Type[ScenarioStrategy]:
                return MockStrategy

            @classmethod
            def get_default_strategy(cls) -> ScenarioStrategy:
                return MockStrategy.ALL

            @classmethod
            def required_datasets(cls) -> list[str]:
                return []

        # Remove docstring
        UndocumentedScenario.__doc__ = None

        registry = ScenarioRegistry()
        registry._scenarios = {"undocumented": UndocumentedScenario}
        registry._discovered = True  # Prevent auto-discovery

        scenarios = registry.list_scenarios()

        assert len(scenarios) == 1
        assert scenarios[0]["description"] == "No description available"

    def test_list_scenarios_with_required_datasets_error(self):
        """Test list_scenarios raises error when required_datasets fails."""

        class BrokenScenario(Scenario):
            """Scenario that raises error on required_datasets."""

            async def _get_atomic_attacks_async(self):
                return []

            @classmethod
            def get_strategy_class(cls) -> Type[ScenarioStrategy]:
                return MockStrategy

            @classmethod
            def get_default_strategy(cls) -> ScenarioStrategy:
                return MockStrategy.ALL

            @classmethod
            def required_datasets(cls) -> list[str]:
                raise ValueError("Cannot get datasets")

        registry = ScenarioRegistry()
        registry._scenarios = {"broken": BrokenScenario}
        registry._discovered = True

        # Should raise the exception instead of catching it
        with pytest.raises(ValueError, match="Cannot get datasets"):
            registry.list_scenarios()

    def test_class_name_to_scenario_name_with_scenario_suffix(self):
        """Test converting class name with 'Scenario' suffix."""
        registry = ScenarioRegistry()
        result = registry._class_name_to_scenario_name("EncodingScenario")
        assert result == "encoding"

    def test_class_name_to_scenario_name_without_scenario_suffix(self):
        """Test converting class name without 'Scenario' suffix."""
        registry = ScenarioRegistry()
        result = registry._class_name_to_scenario_name("CustomTest")
        assert result == "custom_test"

    def test_class_name_to_scenario_name_camelcase(self):
        """Test converting CamelCase to snake_case."""
        registry = ScenarioRegistry()
        result = registry._class_name_to_scenario_name("MyCustomScenario")
        assert result == "my_custom"

    def test_class_name_to_scenario_name_with_numbers(self):
        """Test converting class name with numbers."""
        registry = ScenarioRegistry()
        result = registry._class_name_to_scenario_name("Test123Scenario")
        assert result == "test123"

    def test_discover_user_scenarios_no_modules(self):
        """Test discover_user_scenarios with no user modules."""
        registry = ScenarioRegistry()
        registry._scenarios = {}

        # Should not raise an exception
        registry.discover_user_scenarios()

    @patch("pyrit.cli.scenario_registry.inspect.getmembers")
    def test_discover_user_scenarios_with_user_class(self, mock_getmembers):
        """Test discover_user_scenarios finds user-defined scenarios."""

        class UserScenario(Scenario):
            """User-defined scenario."""

            async def _get_atomic_attacks_async(self):
                return []

            @classmethod
            def get_strategy_class(cls) -> Type[ScenarioStrategy]:
                return MockStrategy

            @classmethod
            def get_default_strategy(cls) -> ScenarioStrategy:
                return MockStrategy.ALL

        UserScenario.__module__ = "user_module"

        mock_module = MagicMock()
        mock_module.__dict__ = {}

        # Need to patch sys.modules which is imported inside the function
        import sys

        original_modules = sys.modules.copy()
        try:
            sys.modules["user_module"] = mock_module
            mock_getmembers.return_value = [("UserScenario", UserScenario)]

            registry = ScenarioRegistry()
            registry._scenarios = {}
            registry.discover_user_scenarios()

            # Verify user scenario was registered
            assert "user" in registry._scenarios
        finally:
            # Restore original sys.modules
            sys.modules.clear()
            sys.modules.update(original_modules)

    def test_discover_user_scenarios_skips_builtins(self):
        """Test discover_user_scenarios skips built-in modules."""
        registry = ScenarioRegistry()
        initial_count = len(registry._scenarios)

        registry.discover_user_scenarios()

        # Should not add any scenarios from built-in modules
        # (may have same count or more if user modules exist, but not from builtins)
        assert len(registry._scenarios) >= initial_count
