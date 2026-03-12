# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Literal

LengthMode = Literal["normal", "detailed", "near_max"]


def build_length_mode_instruction(*, length_mode: LengthMode, focus: str) -> str:
    """Return a short instruction block for longer LLM-generated rewrites."""
    if length_mode == "normal":
        return ""

    base_instruction = (
        "Preserve the user's intent and the converter's attack style. "
        "Use the extra space for useful detail and reinforcement, not filler or empty repetition."
    )

    if length_mode == "detailed":
        return (
            f"{base_instruction} "
            f"Make the rewritten output meaningfully longer and more detailed. {focus}"
        )

    return (
        f"{base_instruction} "
        "Make the rewritten output much longer and richly developed when the source material allows. "
        "Aim for roughly 2800 to 3400 characters instead of a short answer. "
        f"{focus}"
    )
