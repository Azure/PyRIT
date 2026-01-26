# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Tests for the Jailbreak class."""

from unittest.mock import patch

import pytest

from pyrit.scenario.scenarios.airt.jailbreak import JailbreakStrategy

# TODO Fixtures


@pytest.fixture
def mock_jailbreaks(): ...


@pytest.fixture
def mock_objective_target(): ...


@pytest.fixture
def mock_objective_scorer(): ...


@pytest.fixture
def all_jailbreak_strategy():
    return JailbreakStrategy.ALL


@pytest.fixture
def pyrit_jailbreak_strategy():
    return JailbreakStrategy.PYRIT


@pytest.fixture
def mock_runtime_env():
    with patch.dict(
        "os.environ",
        {
            "AZURE_OPENAI_GPT4O_UNSAFE_CHAT_ENDPOINT": "https://test.openai.azure.com/",
            "AZURE_OPENAI_GPT4O_UNSAFE_CHAT_KEY": "test-key",
            "AZURE_OPENAI_GPT4O_UNSAFE_CHAT_MODEL": "gpt-4",
            "OPENAI_CHAT_ENDPOINT": "https://test.openai.azure.com/",
            "OPENAI_CHAT_KEY": "test-key",
            "OPENAI_CHAT_MODEL": "gpt-4",
        },
    ):
        yield


# TODO Test classes


@pytest.mark.usefixtures("patch_central_database")
class TestJailbreakInitialization: ...


@pytest.mark.usefixtures("patch_central_database")
class TestJailbreakAtomicAttacks: ...


@pytest.mark.usefixtures("patch_central_database")
class TestJailbreakExecution: ...
