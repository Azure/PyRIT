# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Tests for the Jailbreak class."""

from typing import Dict, List
from unittest.mock import MagicMock, patch

import pytest
import yaml

from pyrit.models import SeedGroup, SeedObjective
from pyrit.prompt_target import PromptTarget
from pyrit.scenario.scenarios.airt.jailbreak import Jailbreak, JailbreakStrategy
from pyrit.score.true_false.true_false_inverter_scorer import TrueFalseInverterScorer

JAILBREAKS = False


@pytest.fixture
def mock_jailbreak_template(tmp_path) -> Dict:
    """Create mock YAML for jailbreak template."""
    contents = """
    ---
    name: Sample
    description: Sample
    authors:
        - Sample Sample
    source: Sample
    parameters:
        - prompt
    data_type: text
    value: >
        Sample. Question: {{ prompt }}
    """
    return yaml.safe_load(contents)


@pytest.fixture
def mock_memory_seed_groups() -> List[SeedGroup]:
    """Create mock seed groups that _get_default_seed_groups() would return."""
    return [
        SeedGroup(seeds=[SeedObjective(value=prompt)])
        for prompt in [
            "sample objective 1",
            "sample objective 2",
            "sample objective 3",
        ]
    ]


@pytest.fixture
def mock_objective_target() -> PromptTarget:
    """Create a mock objective target for testing."""
    mock = MagicMock(spec=PromptTarget)
    mock.get_identifier.return_value = {"__type__": "MockObjectiveTarget", "__module__": "test"}
    return mock


@pytest.fixture
def mock_objective_scorer() -> TrueFalseInverterScorer:
    """Create a mock scorer for testing."""
    mock = MagicMock(spec=TrueFalseInverterScorer)
    mock.get_identifier.return_value = {"__type__": "MockObjectiveScorer", "__module__": "test"}
    return mock


@pytest.fixture
def all_jailbreak_strategy() -> JailbreakStrategy:
    return JailbreakStrategy.ALL


@pytest.fixture
def pyrit_jailbreak_strategy() -> JailbreakStrategy:
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


FIXTURES = ["patch_central_database", "mock_runtime_env"]


@pytest.mark.usefixtures(*FIXTURES)
class TestJailbreakInitialization: ...


@pytest.mark.usefixtures(*FIXTURES)
class TestJailbreakAtomicAttacks: ...


@pytest.mark.usefixtures(*FIXTURES)
class TestJailbreakExecution: ...


@pytest.mark.usefixtures(*FIXTURES)
class TestJailbreakProperties:
    """Tests for Jailbreak properties."""

    def test_scenario_version_is_set(
        self,
        *,
        mock_objective_scorer: TrueFalseInverterScorer,
        sample_objectives: List[str],
    ) -> None:
        """Test that scenario version is properly set."""
        scenario = Jailbreak(
            objectives=sample_objectives,
            objective_scorer=mock_objective_scorer,
        )

        assert scenario.version == 1

    @pytest.mark.asyncio
    async def test_no_target_duplication_async(
        self, *, mock_objective_target: PromptTarget, mock_memory_seed_groups: List[SeedGroup]
    ) -> None:
        """Test that all three targets (adversarial, object, scorer) are distinct."""
        with patch.object(Jailbreak, "_resolve_seed_groups", return_value=mock_memory_seed_groups):
            scenario = Jailbreak()
            await scenario.initialize_async(objective_target=mock_objective_target)

            objective_target = scenario._objective_target
            scorer_target = scenario._scorer_config.objective_scorer  # type: ignore

            assert objective_target != scorer_target
