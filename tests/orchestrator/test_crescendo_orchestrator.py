# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import pathlib
import pytest

from typing import Generator

from pyrit.memory.memory_interface import MemoryInterface
from pyrit.prompt_target import AzureOpenAIChatTarget
from pyrit.models import AttackStrategy
from pyrit.common.path import DATASETS_PATH

from tests.mocks import get_memory_interface


@pytest.fixture
def memory_interface() -> Generator[MemoryInterface, None, None]:
    yield from get_memory_interface()


@pytest.fixture
def chat_completion_engine() -> AzureOpenAIChatTarget:
    return AzureOpenAIChatTarget(deployment_name="test", endpoint="test", api_key="test")


@pytest.fixture
def prompt_target(memory_interface) -> AzureOpenAIChatTarget:
    return AzureOpenAIChatTarget(
        deployment_name="test",
        endpoint="test",
        api_key="test",
        memory=memory_interface,
    )


def _test_load_crescendo_variant(id: int) -> None:
    attack_strategy = AttackStrategy(
        strategy=pathlib.Path(DATASETS_PATH) / "orchestrators" / "crescendo" / f"variant_{id}.yaml",
        conversation_objective="1",
        previous_questions_and_summaries_and_scores="2",
        last_response="3",
        current_round="4",
        success_flag="5",
    )

    assert attack_strategy.strategy.name == f"Crescendo Variant {id}"
    assert attack_strategy.kwargs["conversation_objective"] == "1"
    assert attack_strategy.kwargs["previous_questions_and_summaries_and_scores"] == "2"
    assert str(attack_strategy)


def test_load_crescendo_variant_1() -> None:
    _test_load_crescendo_variant(1)


def test_load_crescendo_variant_2() -> None:
    _test_load_crescendo_variant(2)


def test_load_crescendo_variant_3() -> None:
    _test_load_crescendo_variant(3)


def test_load_crescendo_variant_4() -> None:
    _test_load_crescendo_variant(4)


def test_load_crescendo_variant_5() -> None:
    _test_load_crescendo_variant(5)
