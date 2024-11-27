# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import pytest
from pyrit.prompt_converter.claim_converter import prompt_openai

def test_make_prompt_basic():
    instance = "example instance"
    instruction = "This is an instruction."
    result = prompt_openai.make_prompt(instance, instruction)
    expected = "This is an instruction.\n-------\nexample instance->"
    assert result == expected

def test_make_prompt_with_exemplars():
    instance = "example instance"
    instruction = "This is an instruction."
    few_shot_exemplars = {
        "input1": "output1",
        "input2": ["output2a", "output2b"]
    }
    result = prompt_openai.make_prompt(instance, instruction, few_shot_exemplars)
    assert "This is an instruction.\n-------\n" in result
    assert "input1->output1" in result
    assert "example instance->" in result

def test_make_prompt_with_seed():
    instance = "example instance"
    instruction = "This is an instruction."
    few_shot_exemplars = {
        "input1": "output1",
        "input2": ["output2a", "output2b"]
    }
    result1 = prompt_openai.make_prompt(instance, instruction, few_shot_exemplars, seed=42)
    result2 = prompt_openai.make_prompt(instance, instruction, few_shot_exemplars, seed=42)
    assert result1 == result2

def test_make_prompt_with_sample_exemplars():
    instance = "example instance"
    instruction = "This is an instruction."
    few_shot_exemplars = {
        "input1": "output1",
        "input2": ["output2a", "output2b"],
        "input3": "output3"
    }
    result = prompt_openai.make_prompt(instance, instruction, few_shot_exemplars, sample_exemplars=2)
    assert "This is an instruction.\n-------\n" in result
    assert "example instance->" in result
    assert result.count("->") <= 3  # 2 exemplars + instance