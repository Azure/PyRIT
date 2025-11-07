# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Tests for the scenario signature enforcer metaclass."""

from typing import Dict, List, Optional, Sequence
from unittest.mock import MagicMock

import pytest

from pyrit.common import apply_defaults
from pyrit.prompt_target import PromptTarget
from pyrit.scenarios import AtomicAttack, Scenario
from pyrit.scenarios.scenario_signature_enforcer import (
    ScenarioSignatureEnforcer,
    validate_scenario_signature,
)
from pyrit.scenarios.scenario_strategy import ScenarioCompositeStrategy, ScenarioStrategy


@pytest.mark.usefixtures("patch_central_database")
class TestScenarioSignatureEnforcer:
    """Test suite for the ScenarioSignatureEnforcer metaclass."""

    def test_valid_scenario_signature_passes_validation(self):
        """Test that a properly structured scenario passes validation."""

        # This should not raise any exceptions
        class ValidScenario(Scenario):
            """A valid scenario with all required parameters."""

            version: int = 1

            @apply_defaults
            def __init__(
                self,
                *,
                objective_target: PromptTarget,
                scenario_strategies: Optional[Sequence[ScenarioStrategy | ScenarioCompositeStrategy]] = None,
                max_concurrency: int = 10,
                max_retries: int = 0,
                memory_labels: Optional[Dict[str, str]] = None,
                custom_param: str = "test",
            ) -> None:
                """Initialize the valid scenario."""
                super().__init__(
                    name="Valid Scenario",
                    version=self.version,
                    objective_target=objective_target,
                    max_concurrency=max_concurrency,
                    max_retries=max_retries,
                    memory_labels=memory_labels,
                )

            async def _get_atomic_attacks_async(self) -> List[AtomicAttack]:
                """Return empty list for testing."""
                return []

        # If we can create the class without errors, validation passed
        assert ValidScenario is not None

    def test_missing_objective_target_raises_type_error(self):
        """Test that missing objective_target parameter raises TypeError."""

        with pytest.raises(TypeError) as exc_info:

            class MissingObjectiveTargetScenario(Scenario):
                """A scenario missing the objective_target parameter."""

                version: int = 1

                @apply_defaults
                def __init__(
                    self,
                    *,
                    my_target: PromptTarget,  # Wrong name!
                    scenario_strategies: Optional[Sequence[ScenarioStrategy]] = None,
                    max_concurrency: int = 10,
                    max_retries: int = 0,
                    memory_labels: Optional[Dict[str, str]] = None,
                ) -> None:
                    """Initialize the scenario."""
                    super().__init__(
                        name="Bad Scenario",
                        version=self.version,
                        objective_target=my_target,
                        max_concurrency=max_concurrency,
                        max_retries=max_retries,
                        memory_labels=memory_labels,
                    )

                async def _get_atomic_attacks_async(self) -> List[AtomicAttack]:
                    """Return empty list for testing."""
                    return []

        # Verify the error message is helpful
        assert "MissingObjectiveTargetScenario.__init__() missing required parameter(s)" in str(exc_info.value)
        assert "objective_target" in str(exc_info.value)
        assert "CLI compatibility" in str(exc_info.value)

    def test_missing_scenario_strategies_raises_type_error(self):
        """Test that missing scenario_strategies parameter raises TypeError."""

        with pytest.raises(TypeError) as exc_info:

            class MissingStrategiesScenario(Scenario):
                """A scenario missing the scenario_strategies parameter."""

                version: int = 1

                @apply_defaults
                def __init__(
                    self,
                    *,
                    objective_target: PromptTarget,
                    max_concurrency: int = 10,
                    max_retries: int = 0,
                    memory_labels: Optional[Dict[str, str]] = None,
                ) -> None:
                    """Initialize the scenario."""
                    super().__init__(
                        name="Bad Scenario",
                        version=self.version,
                        objective_target=objective_target,
                        max_concurrency=max_concurrency,
                        max_retries=max_retries,
                        memory_labels=memory_labels,
                    )

                async def _get_atomic_attacks_async(self) -> List[AtomicAttack]:
                    """Return empty list for testing."""
                    return []

        assert "MissingStrategiesScenario.__init__() missing required parameter(s)" in str(exc_info.value)
        assert "scenario_strategies" in str(exc_info.value)

    def test_missing_max_concurrency_raises_type_error(self):
        """Test that missing max_concurrency parameter raises TypeError."""

        with pytest.raises(TypeError) as exc_info:

            class MissingMaxConcurrencyScenario(Scenario):
                """A scenario missing the max_concurrency parameter."""

                version: int = 1

                @apply_defaults
                def __init__(
                    self,
                    *,
                    objective_target: PromptTarget,
                    scenario_strategies: Optional[Sequence[ScenarioStrategy]] = None,
                    max_retries: int = 0,
                    memory_labels: Optional[Dict[str, str]] = None,
                ) -> None:
                    """Initialize the scenario."""
                    super().__init__(
                        name="Bad Scenario",
                        version=self.version,
                        objective_target=objective_target,
                        max_concurrency=5,
                        max_retries=max_retries,
                        memory_labels=memory_labels,
                    )

                async def _get_atomic_attacks_async(self) -> List[AtomicAttack]:
                    """Return empty list for testing."""
                    return []

        assert "MissingMaxConcurrencyScenario.__init__() missing required parameter(s)" in str(exc_info.value)
        assert "max_concurrency" in str(exc_info.value)

    def test_missing_max_retries_raises_type_error(self):
        """Test that missing max_retries parameter raises TypeError."""

        with pytest.raises(TypeError) as exc_info:

            class MissingMaxRetriesScenario(Scenario):
                """A scenario missing the max_retries parameter."""

                version: int = 1

                @apply_defaults
                def __init__(
                    self,
                    *,
                    objective_target: PromptTarget,
                    scenario_strategies: Optional[Sequence[ScenarioStrategy]] = None,
                    max_concurrency: int = 10,
                    memory_labels: Optional[Dict[str, str]] = None,
                ) -> None:
                    """Initialize the scenario."""
                    super().__init__(
                        name="Bad Scenario",
                        version=self.version,
                        objective_target=objective_target,
                        max_concurrency=max_concurrency,
                        max_retries=0,
                        memory_labels=memory_labels,
                    )

                async def _get_atomic_attacks_async(self) -> List[AtomicAttack]:
                    """Return empty list for testing."""
                    return []

        assert "MissingMaxRetriesScenario.__init__() missing required parameter(s)" in str(exc_info.value)
        assert "max_retries" in str(exc_info.value)

    def test_missing_memory_labels_raises_type_error(self):
        """Test that missing memory_labels parameter raises TypeError."""

        with pytest.raises(TypeError) as exc_info:

            class MissingMemoryLabelsScenario(Scenario):
                """A scenario missing the memory_labels parameter."""

                version: int = 1

                @apply_defaults
                def __init__(
                    self,
                    *,
                    objective_target: PromptTarget,
                    scenario_strategies: Optional[Sequence[ScenarioStrategy]] = None,
                    max_concurrency: int = 10,
                    max_retries: int = 0,
                ) -> None:
                    """Initialize the scenario."""
                    super().__init__(
                        name="Bad Scenario",
                        version=self.version,
                        objective_target=objective_target,
                        max_concurrency=max_concurrency,
                        max_retries=max_retries,
                        memory_labels=None,
                    )

                async def _get_atomic_attacks_async(self) -> List[AtomicAttack]:
                    """Return empty list for testing."""
                    return []

        assert "MissingMemoryLabelsScenario.__init__() missing required parameter(s)" in str(exc_info.value)
        assert "memory_labels" in str(exc_info.value)

    def test_missing_multiple_parameters_raises_type_error(self):
        """Test that missing multiple parameters shows all missing params in error."""

        with pytest.raises(TypeError) as exc_info:

            class MissingMultipleParamsScenario(Scenario):
                """A scenario missing multiple required parameters."""

                version: int = 1

                @apply_defaults
                def __init__(
                    self,
                    *,
                    objective_target: PromptTarget,
                    max_concurrency: int = 10,
                ) -> None:
                    """Initialize the scenario."""
                    super().__init__(
                        name="Bad Scenario",
                        version=self.version,
                        objective_target=objective_target,
                        max_concurrency=max_concurrency,
                        max_retries=0,
                        memory_labels=None,
                    )

                async def _get_atomic_attacks_async(self) -> List[AtomicAttack]:
                    """Return empty list for testing."""
                    return []

        error_message = str(exc_info.value)
        assert "MissingMultipleParamsScenario.__init__() missing required parameter(s)" in error_message
        # Should show all missing parameters
        assert "scenario_strategies" in error_message
        assert "max_retries" in error_message
        assert "memory_labels" in error_message

    def test_scenario_with_additional_parameters_passes(self):
        """Test that scenarios can have additional custom parameters beyond required ones."""

        class ScenarioWithExtras(Scenario):
            """A scenario with extra parameters."""

            version: int = 1

            @apply_defaults
            def __init__(
                self,
                *,
                objective_target: PromptTarget,
                scenario_strategies: Optional[Sequence[ScenarioStrategy]] = None,
                max_concurrency: int = 10,
                max_retries: int = 0,
                memory_labels: Optional[Dict[str, str]] = None,
                custom_param1: str = "test1",
                custom_param2: int = 42,
                custom_param3: Optional[list] = None,
            ) -> None:
                """Initialize the scenario with extra parameters."""
                super().__init__(
                    name="Scenario With Extras",
                    version=self.version,
                    objective_target=objective_target,
                    max_concurrency=max_concurrency,
                    max_retries=max_retries,
                    memory_labels=memory_labels,
                )

            async def _get_atomic_attacks_async(self) -> List[AtomicAttack]:
                """Return empty list for testing."""
                return []

        # Should not raise any exceptions
        assert ScenarioWithExtras is not None

    def test_abstract_scenario_skips_validation(self):
        """Test that abstract scenarios are not validated."""

        # This should not raise exceptions even though it's missing parameters
        class AbstractTestScenario(Scenario):
            """An abstract scenario."""

            __abstract__ = True  # Mark as abstract

            def __init__(self, *, some_param: str) -> None:
                """Initialize with non-standard parameters."""
                pass

            async def _get_atomic_attacks_async(self) -> List[AtomicAttack]:
                """Return empty list for testing."""
                return []

        # Should not raise because it's marked as abstract
        assert AbstractTestScenario is not None

    def test_validate_scenario_signature_function_with_valid_class(self):
        """Test the validate_scenario_signature function with a valid scenario."""

        class ValidScenarioForFunction(Scenario):
            """A valid scenario."""

            version: int = 1

            @apply_defaults
            def __init__(
                self,
                *,
                objective_target: PromptTarget,
                scenario_strategies: Optional[Sequence[ScenarioStrategy]] = None,
                max_concurrency: int = 10,
                max_retries: int = 0,
                memory_labels: Optional[Dict[str, str]] = None,
            ) -> None:
                """Initialize the scenario."""
                super().__init__(
                    name="Valid Scenario",
                    version=self.version,
                    objective_target=objective_target,
                    max_concurrency=max_concurrency,
                    max_retries=max_retries,
                    memory_labels=memory_labels,
                )

            async def _get_atomic_attacks_async(self) -> List[AtomicAttack]:
                """Return empty list for testing."""
                return []

        # Should not raise any exceptions
        validate_scenario_signature(ValidScenarioForFunction)

    def test_validate_scenario_signature_function_with_invalid_class(self):
        """Test the validate_scenario_signature function with an invalid scenario."""

        # Mark as abstract to bypass metaclass validation during class creation
        # so we can test the manual validation function
        class InvalidScenarioForFunction(Scenario):
            """An invalid scenario."""

            __abstract__ = True  # Skip metaclass validation
            version: int = 1

            @apply_defaults
            def __init__(
                self,
                *,
                objective_target: PromptTarget,
                max_concurrency: int = 10,
            ) -> None:
                """Initialize the scenario."""
                super().__init__(
                    name="Invalid Scenario",
                    version=self.version,
                    objective_target=objective_target,
                    max_concurrency=max_concurrency,
                    max_retries=0,
                    memory_labels=None,
                )

            async def _get_atomic_attacks_async(self) -> List[AtomicAttack]:
                """Return empty list for testing."""
                return []

        # Manual validation should raise TypeError
        with pytest.raises(TypeError) as exc_info:
            validate_scenario_signature(InvalidScenarioForFunction)

        assert "InvalidScenarioForFunction.__init__() missing required parameter(s)" in str(exc_info.value)

    def test_error_message_includes_cli_example(self):
        """Test that error messages include helpful CLI usage examples."""

        with pytest.raises(TypeError) as exc_info:

            class BadScenarioForCliExample(Scenario):
                """A scenario for testing error message quality."""

                version: int = 1

                @apply_defaults
                def __init__(
                    self,
                    *,
                    objective_target: PromptTarget,
                    max_concurrency: int = 10,
                    max_retries: int = 0,
                    memory_labels: Optional[Dict[str, str]] = None,
                ) -> None:
                    """Initialize the scenario."""
                    super().__init__(
                        name="Bad Scenario",
                        version=self.version,
                        objective_target=objective_target,
                        max_concurrency=max_concurrency,
                        max_retries=max_retries,
                        memory_labels=memory_labels,
                    )

                async def _get_atomic_attacks_async(self) -> List[AtomicAttack]:
                    """Return empty list for testing."""
                    return []

        error_message = str(exc_info.value)
        # Should include a CLI usage example
        assert "pyrit run" in error_message or "CLI" in error_message
        assert "--objective_target" in error_message
        assert "--scenario_strategies" in error_message

    def test_scenario_without_init_skips_validation(self):
        """Test that scenarios without their own __init__ are not validated."""

        # This scenario doesn't define its own __init__, so it won't be validated
        class ScenarioWithoutInit(Scenario):
            """A scenario that inherits __init__ from parent."""

            version: int = 1

            async def _get_atomic_attacks_async(self) -> List[AtomicAttack]:
                """Return empty list for testing."""
                return []

        # Should not raise because it doesn't define __init__
        assert ScenarioWithoutInit is not None

    def test_required_parameters_constant_is_complete(self):
        """Test that the REQUIRED_PARAMETERS constant includes all expected parameters."""
        required_params = ScenarioSignatureEnforcer.REQUIRED_PARAMETERS

        # Verify all required parameters are present
        assert "objective_target" in required_params
        assert "scenario_strategies" in required_params
        assert "max_concurrency" in required_params
        assert "max_retries" in required_params
        assert "memory_labels" in required_params

        # Verify we have exactly 5 required parameters
        assert len(required_params) == 5

    def test_objective_target_can_have_default_value(self):
        """Test that objective_target can have a default value (for @apply_defaults)."""

        # This should NOT raise an error - we allow default values now
        class ScenarioWithDefaultTarget(Scenario):
            """A scenario with default objective_target."""

            version: int = 1

            @apply_defaults
            def __init__(
                self,
                *,
                objective_target: Optional[PromptTarget] = None,
                scenario_strategies: Optional[Sequence[ScenarioStrategy]] = None,
                max_concurrency: int = 10,
                max_retries: int = 0,
                memory_labels: Optional[Dict[str, str]] = None,
            ) -> None:
                """Initialize the scenario."""
                super().__init__(
                    name="Scenario With Default",
                    version=self.version,
                    objective_target=objective_target,
                    max_concurrency=max_concurrency,
                    max_retries=max_retries,
                    memory_labels=memory_labels,
                )

            async def _get_atomic_attacks_async(self) -> List[AtomicAttack]:
                """Return empty list for testing."""
                return []

        # Should not raise - default values are allowed
        assert ScenarioWithDefaultTarget is not None
