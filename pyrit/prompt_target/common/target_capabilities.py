# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from dataclasses import dataclass, field, fields

from pyrit.models import PromptDataType


@dataclass(frozen=True)
class TargetCapabilities:
    """
    Describes the capabilities of a PromptTarget so that attacks
    and other components can adapt their behavior accordingly.

    Each target class defines default capabilities via the _DEFAULT_CAPABILITIES
    class attribute. Users can override individual capabilities per instance
    through constructor parameters, which is useful for targets whose
    capabilities depend on deployment configuration (e.g., Playwright, HTTP).
    """

    # Whether the target natively supports multi-turn conversations
    # (i.e., it accepts and uses conversation history or maintains state
    # across turns via external mechanisms like WebSocket connections).
    supports_multi_turn: bool = False

    # Whether the target natively supports multiple message pieces in a single request.
    supports_multi_message_pieces: bool = True

    # Whether the target natively supports JSON output (e.g., via a "json" response format).
    supports_json_response: bool = False

    # The input modalities supported by the target (e.g., "text", "image").
    input_modalities: list[PromptDataType] = field(default_factory=lambda: ["text"])

    # The output modalities supported by the target (e.g., "text", "image").
    output_modalities: list[PromptDataType] = field(default_factory=lambda: ["text"])

    def assert_satifies(self, required_capabilities: "TargetCapabilities") -> None:
        """
        Assert that the current capabilities satisfy the required capabilities.

        Args:
            required_capabilities (TargetCapabilities): The required capabilities to check against.

        Raises:
            ValueError: If any of the required capabilities are not satisfied.
        """
        unmet = []
        for f in fields(required_capabilities):
            required_value = getattr(required_capabilities, f.name)
            self_value = getattr(self, f.name)
            if isinstance(required_value, list):
                missing = set(required_value) - set(self_value)
                if missing:
                    unmet.append(f"{f.name}: missing {missing}")
            elif required_value and not self_value:
                unmet.append(f.name)
        if unmet:
            raise ValueError(f"Target does not satisfy the following capabilities: {', '.join(unmet)}")
        

