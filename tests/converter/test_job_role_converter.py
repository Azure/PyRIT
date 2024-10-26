# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
from unittest.mock import AsyncMock

from pyrit.prompt_converter.job_role_converter import JobRoleGenerator


# Test manual prompt list for JobRoleGenerator
def test_job_role_generator_manual_prompt_list() -> None:
    mock_converter_target = AsyncMock()

    converter = JobRoleGenerator(converter_target=mock_converter_target)
    job = "job role"
    manual_prompts = converter.manual_attack_demographics(job)
    assert len(manual_prompts) == 8
    assert manual_prompts == [
        "job role (Male)",
        "job role (Female)",
        "job role (Malay)",
        "job role (Chinese)",
        "job role (Indian)",
        "job role (White)",
        "job role (Black)",
        "job role (African-American)",
    ]

# Test input supported for JobRoleGenerator
def test_job_role_generator_input_supported() -> None:
    mock_converter_target = AsyncMock()

    converter = JobRoleGenerator(converter_target=mock_converter_target)
    assert converter.input_supported("text") is True
    assert converter.input_supported("image") is False
