# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Type, TypeVar

from pyrit.models import Message, SeedGroup

AttackParamsT = TypeVar("AttackParamsT", bound="AttackParameters")


@dataclass(frozen=True)
class AttackParameters:
    """
    Immutable parameters for attack execution.

    This class defines the standard contract for attack parameters. All attacks
    at a given level of the hierarchy share the same parameter signature.

    Attacks that don't accept certain parameters should use the `excluding()` factory
    to create a derived params type without those fields. Attacks that need additional
    parameters should extend this class with new fields.
    """

    # Natural-language description of what the attack tries to achieve (required)
    objective: str

    # Optional message to send to the objective target (overrides objective if provided)
    next_message: Optional[Message] = None

    # Conversation that is automatically prepended to the target model
    prepended_conversation: Optional[List[Message]] = None

    # Additional labels that can be applied to the prompts throughout the attack
    memory_labels: Optional[Dict[str, str]] = field(default_factory=dict)

    @classmethod
    def from_seed_group(
        cls: Type[AttackParamsT],
        seed_group: SeedGroup,
        **overrides: Any,
    ) -> AttackParamsT:
        """
        Create an AttackParameters instance from a SeedGroup.

        Extracts standard fields from the seed group and applies any overrides.
        Raises ValueError if overrides contain fields not accepted by this params type.

        Args:
            seed_group: The seed group to extract parameters from.
            **overrides: Field overrides to apply. Must be valid fields for this params type.

        Returns:
            An instance of this AttackParameters type.

        Raises:
            ValueError: If seed_group has no objective or if overrides contain invalid fields.
        """
        # Get valid field names for this params type
        valid_fields = {f.name for f in dataclasses.fields(cls)}

        # Validate overrides don't contain invalid fields
        invalid_fields = set(overrides.keys()) - valid_fields
        if invalid_fields:
            raise ValueError(
                f"{cls.__name__} does not accept parameters: {invalid_fields}. "
                f"Accepted parameters: {valid_fields}"
            )

        # Extract objective (required)
        if seed_group.objective is None:
            raise ValueError("SeedGroup must have an objective")

        # Build params dict, only including fields this class accepts
        params: Dict[str, Any] = {}

        if "objective" in valid_fields:
            params["objective"] = seed_group.objective.value

        if "next_message" in valid_fields:
            params["next_message"] = seed_group.next_message

        if "prepended_conversation" in valid_fields:
            params["prepended_conversation"] = seed_group.prepended_conversation

        if "memory_labels" in valid_fields:
            params["memory_labels"] = {}

        # Apply overrides (already validated above)
        params.update(overrides)

        return cls(**params)

    @classmethod
    def excluding(cls, *field_names: str) -> Type["AttackParameters"]:
        """
        Create a new AttackParameters subclass that excludes the specified fields.

        This factory method creates a frozen dataclass without the specified fields.
        The resulting class inherits the `from_seed_group()` behavior and will raise
        if excluded fields are passed as overrides.

        Args:
            *field_names: Names of fields to exclude from the new params type.

        Returns:
            A new AttackParameters subclass without the specified fields.

        Raises:
            ValueError: If any field_name is not a valid field of this class.

        Example:
            RolePlayAttackParameters = AttackParameters.excluding("next_message", "prepended_conversation")
        """
        # Validate all field names exist
        current_fields = {f.name for f in dataclasses.fields(cls)}
        invalid = set(field_names) - current_fields
        if invalid:
            raise ValueError(f"Cannot exclude non-existent fields: {invalid}. Valid fields: {current_fields}")

        # Build new fields list excluding the specified ones
        new_fields: List[tuple] = []
        for f in dataclasses.fields(cls):
            if f.name not in field_names:
                # Preserve field defaults
                if f.default is not dataclasses.MISSING:
                    new_fields.append((f.name, f.type, field(default=f.default)))
                elif f.default_factory is not dataclasses.MISSING:
                    new_fields.append((f.name, f.type, field(default_factory=f.default_factory)))
                else:
                    new_fields.append((f.name, f.type))

        # Generate a descriptive class name
        excluded_str = "_".join(sorted(field_names))
        class_name = f"{cls.__name__}Excluding_{excluded_str}"

        # Create the new dataclass
        new_cls = dataclasses.make_dataclass(
            class_name,
            new_fields,
            frozen=True,
        )

        # Copy the from_seed_group method to the new class
        # We need to bind it as a classmethod on the new class
        new_cls.from_seed_group = classmethod(lambda c, sg, **ov: cls._from_seed_group_impl(c, sg, **ov))

        return new_cls

    @classmethod
    def _from_seed_group_impl(
        cls: Type[AttackParamsT],
        target_cls: Type[AttackParamsT],
        seed_group: SeedGroup,
        **overrides: Any,
    ) -> AttackParamsT:
        """
        Implementation of from_seed_group that can be used by dynamically created classes.

        Args:
            target_cls: The actual class to instantiate (may be a dynamically created subclass).
            seed_group: The seed group to extract parameters from.
            **overrides: Field overrides to apply.

        Returns:
            An instance of target_cls.
        """
        # Get valid field names for the target class
        valid_fields = {f.name for f in dataclasses.fields(target_cls)}

        # Validate overrides don't contain invalid fields
        invalid_fields = set(overrides.keys()) - valid_fields
        if invalid_fields:
            raise ValueError(
                f"{target_cls.__name__} does not accept parameters: {invalid_fields}. "
                f"Accepted parameters: {valid_fields}"
            )

        # Extract objective (required)
        if seed_group.objective is None:
            raise ValueError("SeedGroup must have an objective")

        # Build params dict, only including fields the target class accepts
        params: Dict[str, Any] = {}

        if "objective" in valid_fields:
            params["objective"] = seed_group.objective.value

        if "next_message" in valid_fields:
            params["next_message"] = seed_group.next_message

        if "prepended_conversation" in valid_fields:
            params["prepended_conversation"] = seed_group.prepended_conversation

        if "memory_labels" in valid_fields:
            params["memory_labels"] = {}

        # Apply overrides (already validated above)
        params.update(overrides)

        return target_cls(**params)
