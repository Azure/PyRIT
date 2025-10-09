# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Optional

import pytest

from pyrit.setup import apply_defaults, get_global_default_values, set_default_value
from pyrit.setup.pyrit_default_value import DefaultValueScope


class TestApplyDefaultsDecorator:
    """Tests for the @apply_defaults decorator."""

    def setup_method(self) -> None:
        """Clear any existing default values before each test."""
        get_global_default_values()._default_values.clear()

    def test_no_defaults_configured_returns_none(self) -> None:
        """Test that parameters remain None when no defaults are configured."""

        class TestClass:
            @apply_defaults
            def __init__(self, *, param1: Optional[str] = None, param2: Optional[int] = None) -> None:
                self.param1 = param1
                self.param2 = param2

        obj = TestClass()
        assert obj.param1 is None
        assert obj.param2 is None

    def test_single_default_value_applied(self) -> None:
        """Test that a single default value is applied correctly."""

        class TestClass:
            @apply_defaults
            def __init__(self, *, param1: Optional[str] = None) -> None:
                self.param1 = param1

        set_default_value(class_type=TestClass, parameter_name="param1", value="default_value")

        obj = TestClass()
        assert obj.param1 == "default_value"

    def test_multiple_default_values_applied(self) -> None:
        """Test that multiple default values are applied correctly."""

        class TestClass:
            @apply_defaults
            def __init__(
                self, *, param1: Optional[str] = None, param2: Optional[int] = None, param3: Optional[float] = None
            ) -> None:
                self.param1 = param1
                self.param2 = param2
                self.param3 = param3

        set_default_value(class_type=TestClass, parameter_name="param1", value="test")
        set_default_value(class_type=TestClass, parameter_name="param2", value=42)
        set_default_value(class_type=TestClass, parameter_name="param3", value=3.14)

        obj = TestClass()
        assert obj.param1 == "test"
        assert obj.param2 == 42
        assert obj.param3 == 3.14

    def test_explicit_value_overrides_default(self) -> None:
        """Test that explicitly provided values override defaults."""

        class TestClass:
            @apply_defaults
            def __init__(self, *, param1: Optional[str] = None, param2: Optional[int] = None) -> None:
                self.param1 = param1
                self.param2 = param2

        set_default_value(class_type=TestClass, parameter_name="param1", value="default")
        set_default_value(class_type=TestClass, parameter_name="param2", value=100)

        obj = TestClass(param1="explicit", param2=200)
        assert obj.param1 == "explicit"
        assert obj.param2 == 200

    def test_partial_override_uses_remaining_defaults(self) -> None:
        """Test that overriding some values still uses defaults for others."""

        class TestClass:
            @apply_defaults
            def __init__(
                self, *, param1: Optional[str] = None, param2: Optional[int] = None, param3: Optional[float] = None
            ) -> None:
                self.param1 = param1
                self.param2 = param2
                self.param3 = param3

        set_default_value(class_type=TestClass, parameter_name="param1", value="default1")
        set_default_value(class_type=TestClass, parameter_name="param2", value=100)
        set_default_value(class_type=TestClass, parameter_name="param3", value=1.5)

        obj = TestClass(param2=200)
        assert obj.param1 == "default1"
        assert obj.param2 == 200
        assert obj.param3 == 1.5

    def test_falsy_values_are_not_overridden(self) -> None:
        """Test that falsy values (0, False, "") are preserved and not treated as None."""

        class TestClass:
            @apply_defaults
            def __init__(
                self,
                *,
                param_int: Optional[int] = None,
                param_bool: Optional[bool] = None,
                param_str: Optional[str] = None,
            ) -> None:
                self.param_int = param_int
                self.param_bool = param_bool
                self.param_str = param_str

        set_default_value(class_type=TestClass, parameter_name="param_int", value=100)
        set_default_value(class_type=TestClass, parameter_name="param_bool", value=True)
        set_default_value(class_type=TestClass, parameter_name="param_str", value="default")

        obj = TestClass(param_int=0, param_bool=False, param_str="")
        assert obj.param_int == 0
        assert obj.param_bool is False
        assert obj.param_str == ""


class TestInheritance:
    """Tests for default value inheritance behavior."""

    def setup_method(self) -> None:
        """Clear any existing default values before each test."""
        get_global_default_values()._default_values.clear()

    def test_subclass_inherits_parent_defaults(self) -> None:
        """Test that subclass inherits defaults from parent class."""

        class ParentClass:
            @apply_defaults
            def __init__(self, *, param1: Optional[str] = None, param2: Optional[int] = None) -> None:
                self.param1 = param1
                self.param2 = param2

        class ChildClass(ParentClass):
            @apply_defaults
            def __init__(self, *, param1: Optional[str] = None, param2: Optional[int] = None) -> None:
                super().__init__(param1=param1, param2=param2)

        set_default_value(class_type=ParentClass, parameter_name="param1", value="parent_value")
        set_default_value(class_type=ParentClass, parameter_name="param2", value=42)

        child_obj = ChildClass()
        assert child_obj.param1 == "parent_value"
        assert child_obj.param2 == 42

    def test_subclass_specific_defaults_override_parent(self) -> None:
        """Test that subclass-specific defaults take precedence over parent defaults."""

        class ParentClass:
            @apply_defaults
            def __init__(self, *, param1: Optional[str] = None, param2: Optional[int] = None) -> None:
                self.param1 = param1
                self.param2 = param2

        class ChildClass(ParentClass):
            @apply_defaults
            def __init__(self, *, param1: Optional[str] = None, param2: Optional[int] = None) -> None:
                super().__init__(param1=param1, param2=param2)

        set_default_value(class_type=ParentClass, parameter_name="param1", value="parent_value")
        set_default_value(class_type=ParentClass, parameter_name="param2", value=100)

        set_default_value(class_type=ChildClass, parameter_name="param1", value="child_value")

        child_obj = ChildClass()
        assert child_obj.param1 == "child_value"  # Child overrides parent
        assert child_obj.param2 == 100  # Inherited from parent

    def test_multiple_inheritance_levels(self) -> None:
        """Test defaults work correctly with multiple levels of inheritance."""

        class GrandParent:
            @apply_defaults
            def __init__(self, *, param1: Optional[str] = None) -> None:
                self.param1 = param1

        class Parent(GrandParent):
            @apply_defaults
            def __init__(self, *, param1: Optional[str] = None, param2: Optional[int] = None) -> None:
                super().__init__(param1=param1)
                self.param2 = param2

        class Child(Parent):
            @apply_defaults
            def __init__(
                self, *, param1: Optional[str] = None, param2: Optional[int] = None, param3: Optional[float] = None
            ) -> None:
                super().__init__(param1=param1, param2=param2)
                self.param3 = param3

        set_default_value(class_type=GrandParent, parameter_name="param1", value="grandparent")
        set_default_value(class_type=Parent, parameter_name="param2", value=50)
        set_default_value(class_type=Child, parameter_name="param3", value=3.14)

        child_obj = Child()
        assert child_obj.param1 == "grandparent"
        assert child_obj.param2 == 50
        assert child_obj.param3 == 3.14

    def test_parent_not_affected_by_child_defaults(self) -> None:
        """Test that setting defaults on child class doesn't affect parent instances."""

        class ParentClass:
            @apply_defaults
            def __init__(self, *, param1: Optional[str] = None) -> None:
                self.param1 = param1

        class ChildClass(ParentClass):
            @apply_defaults
            def __init__(self, *, param1: Optional[str] = None) -> None:
                super().__init__(param1=param1)

        set_default_value(class_type=ChildClass, parameter_name="param1", value="child_value")

        parent_obj = ParentClass()
        assert parent_obj.param1 is None  # Parent doesn't get child's default


class TestDefaultValueScope:
    """Tests for DefaultValueScope dataclass."""

    def test_default_value_scope_is_hashable(self) -> None:
        """Test that DefaultValueScope can be used as a dictionary key."""

        class TestClass:
            pass

        scope1 = DefaultValueScope(parameter_name="test", class_type=TestClass)
        scope2 = DefaultValueScope(parameter_name="test", class_type=TestClass)

        # Should be able to use as dict key
        test_dict = {scope1: "value1"}
        assert test_dict[scope2] == "value1"  # Same scope should retrieve same value

    def test_default_value_scope_equality(self) -> None:
        """Test that scopes with same values are equal."""

        class TestClass:
            pass

        scope1 = DefaultValueScope(parameter_name="test", class_type=TestClass)
        scope2 = DefaultValueScope(parameter_name="test", class_type=TestClass)

        assert scope1 == scope2

    def test_default_value_scope_inequality(self) -> None:
        """Test that scopes with different values are not equal."""

        class TestClass1:
            pass

        class TestClass2:
            pass

        scope1 = DefaultValueScope(parameter_name="test", class_type=TestClass1)
        scope2 = DefaultValueScope(parameter_name="test", class_type=TestClass2)
        scope3 = DefaultValueScope(parameter_name="other", class_type=TestClass1)

        assert scope1 != scope2
        assert scope1 != scope3


class TestPyRITDefaultValues:
    """Tests for the PyRITDefaultValues class."""

    def setup_method(self) -> None:
        """Clear any existing default values before each test."""
        get_global_default_values()._default_values.clear()

    def test_get_default_value_returns_configured_value(self) -> None:
        """Test that get_default_value returns configured values."""

        class TestClass:
            pass

        scope = DefaultValueScope(parameter_name="test", class_type=TestClass)
        defaults = get_global_default_values()
        defaults.set_default_value(scope=scope, value="test_value")

        result = defaults.get_default_value(scope=scope, default_value=None)
        assert result == "test_value"

    def test_get_default_value_returns_fallback_when_not_configured(self) -> None:
        """Test that get_default_value returns fallback when no value is configured."""

        class TestClass:
            pass

        scope = DefaultValueScope(parameter_name="test", class_type=TestClass)
        defaults = get_global_default_values()

        result = defaults.get_default_value(scope=scope, default_value="fallback")
        assert result == "fallback"

    def test_get_default_value_for_parameter_with_provided_value(self) -> None:
        """Test that provided values are returned regardless of defaults."""

        class TestClass:
            pass

        set_default_value(class_type=TestClass, parameter_name="test", value="default")
        defaults = get_global_default_values()

        result = defaults.get_default_value_for_parameter(
            class_type=TestClass, parameter_name="test", provided_value="provided"
        )
        assert result == "provided"

    def test_get_default_value_for_parameter_without_provided_value(self) -> None:
        """Test that default value is returned when provided value is None."""

        class TestClass:
            pass

        set_default_value(class_type=TestClass, parameter_name="test", value="default")
        defaults = get_global_default_values()

        result = defaults.get_default_value_for_parameter(
            class_type=TestClass, parameter_name="test", provided_value=None
        )
        assert result == "default"


class TestSetDefaultValue:
    """Tests for the set_default_value convenience function."""

    def setup_method(self) -> None:
        """Clear any existing default values before each test."""
        get_global_default_values()._default_values.clear()

    def test_set_default_value_stores_value(self) -> None:
        """Test that set_default_value properly stores values."""

        class TestClass:
            @apply_defaults
            def __init__(self, *, param1: Optional[str] = None) -> None:
                self.param1 = param1

        set_default_value(class_type=TestClass, parameter_name="param1", value="stored_value")

        obj = TestClass()
        assert obj.param1 == "stored_value"

    def test_set_default_value_overwrites_existing(self) -> None:
        """Test that setting a default value overwrites any existing value."""

        class TestClass:
            @apply_defaults
            def __init__(self, *, param1: Optional[str] = None) -> None:
                self.param1 = param1

        set_default_value(class_type=TestClass, parameter_name="param1", value="first_value")
        set_default_value(class_type=TestClass, parameter_name="param1", value="second_value")

        obj = TestClass()
        assert obj.param1 == "second_value"


class TestComplexScenarios:
    """Tests for complex real-world scenarios."""

    def setup_method(self) -> None:
        """Clear any existing default values before each test."""
        get_global_default_values()._default_values.clear()

    def test_openai_chat_target_scenario(self) -> None:
        """Test a realistic scenario similar to OpenAIChatTarget."""

        class OpenAIChatTarget:
            @apply_defaults
            def __init__(
                self,
                *,
                temperature: Optional[float] = None,
                top_p: Optional[float] = None,
                max_tokens: Optional[int] = None,
            ) -> None:
                self.temperature = temperature
                self.top_p = top_p
                self.max_tokens = max_tokens

        class AzureOpenAIChatTarget(OpenAIChatTarget):
            @apply_defaults
            def __init__(
                self,
                *,
                temperature: Optional[float] = None,
                top_p: Optional[float] = None,
                max_tokens: Optional[int] = None,
                api_version: Optional[str] = None,
            ) -> None:
                super().__init__(temperature=temperature, top_p=top_p, max_tokens=max_tokens)
                self.api_version = api_version

        # Set defaults for base class
        set_default_value(class_type=OpenAIChatTarget, parameter_name="temperature", value=0.7)
        set_default_value(class_type=OpenAIChatTarget, parameter_name="top_p", value=0.9)

        # Set defaults for subclass (more specific temperature)
        set_default_value(class_type=AzureOpenAIChatTarget, parameter_name="temperature", value=0.3)
        set_default_value(class_type=AzureOpenAIChatTarget, parameter_name="api_version", value="2024-10-21")

        # Test base class
        base_obj = OpenAIChatTarget()
        assert base_obj.temperature == 0.7
        assert base_obj.top_p == 0.9
        assert base_obj.max_tokens is None

        # Test subclass with inheritance
        azure_obj = AzureOpenAIChatTarget()
        assert azure_obj.temperature == 0.3  # More specific default
        assert azure_obj.top_p == 0.9  # Inherited from parent
        assert azure_obj.max_tokens is None  # No default set
        assert azure_obj.api_version == "2024-10-21"  # Subclass-specific default

        # Test with explicit overrides
        custom_obj = AzureOpenAIChatTarget(temperature=0.5, max_tokens=100)
        assert custom_obj.temperature == 0.5  # Explicit override
        assert custom_obj.top_p == 0.9  # Still uses default
        assert custom_obj.max_tokens == 100  # Explicit override
        assert custom_obj.api_version == "2024-10-21"  # Still uses default

    def test_multiple_classes_independent_defaults(self) -> None:
        """Test that multiple classes can have independent default configurations."""

        class ClassA:
            @apply_defaults
            def __init__(self, *, param: Optional[str] = None) -> None:
                self.param = param

        class ClassB:
            @apply_defaults
            def __init__(self, *, param: Optional[str] = None) -> None:
                self.param = param

        set_default_value(class_type=ClassA, parameter_name="param", value="value_a")
        set_default_value(class_type=ClassB, parameter_name="param", value="value_b")

        obj_a = ClassA()
        obj_b = ClassB()

        assert obj_a.param == "value_a"
        assert obj_b.param == "value_b"
