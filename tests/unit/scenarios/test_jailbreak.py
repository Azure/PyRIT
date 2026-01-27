# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Tests for the Jailbreak class."""

from typing import Dict, List
from unittest.mock import MagicMock, patch

import pytest
import yaml

from pyrit.executor.attack.core.attack_config import AttackScoringConfig
from pyrit.executor.attack.single_turn.prompt_sending import PromptSendingAttack
from pyrit.models import SeedGroup, SeedObjective
from pyrit.prompt_target import PromptTarget
from pyrit.scenario.scenarios.airt.jailbreak import Jailbreak, JailbreakStrategy
from pyrit.score.true_false.true_false_inverter_scorer import TrueFalseInverterScorer

JAILBREAKS = False


@pytest.fixture
def mock_jailbreak_template() -> Dict:
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
    mock.get_identifier.return_value = {
        "__type__": "MockObjectiveTarget", "__module__": "test"}
    return mock


@pytest.fixture
def mock_objective_scorer() -> TrueFalseInverterScorer:
    """Create a mock scorer for testing."""
    mock = MagicMock(spec=TrueFalseInverterScorer)
    mock.get_identifier.return_value = {
        "__type__": "MockObjectiveScorer", "__module__": "test"}
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
class TestJailbreakInitialization:
    """Tests for Jailbreak initialization."""

    def test_init_with_custom_objectives(self, mock_objective_scorer, sample_objectives):
        """Test initialization with custom objectives."""

        scenario = Jailbreak(
            objectives=sample_objectives,
            objective_scorer=mock_objective_scorer,
        )

        assert scenario.name == "Jailbreak"
        assert scenario.version == 1

    def test_init_with_default_objectives(self, mock_objective_scorer, mock_memory_seed_groups):
        """Test initialization with default objectives."""

        with patch.object(Jailbreak, "_resolve_seed_groups", return_value=mock_memory_seed_groups):
            scenario = Jailbreak(objective_scorer=mock_objective_scorer)

            assert scenario.name == "Jailbreak"
            assert scenario.version == 1

    def test_init_with_default_scorer(self, mock_memory_seed_groups):
        """Test initialization with default scorer."""
        with patch.object(Jailbreak, "_resolve_seed_groups", return_value=mock_memory_seed_groups):
            scenario = Jailbreak()
            assert scenario._objective_scorer_identifier

    def test_init_with_custom_scorer(self, mock_objective_scorer, mock_memory_seed_groups):
        """Test initialization with custom scorer."""
        with patch.object(Jailbreak, "_resolve_seed_groups", return_value=mock_memory_seed_groups):
            scenario = Jailbreak(objective_scorer=mock_objective_scorer)
            assert isinstance(scenario._scorer_config, AttackScoringConfig)

    def test_init_jailbreak_templating(
        self,
    ):
        """Test that jailbreak templating works."""
        pass

    @pytest.mark.asyncio
    async def test_init_raises_exception_when_no_datasets_available(self, mock_objective_target, mock_objective_scorer):
        """Test that initialization raises ValueError when datasets are not available in memory."""
        # Don't mock _resolve_seed_groups, let it try to load from empty memory
        scenario = Jailbreak(objective_scorer=mock_objective_scorer)

        # Error should occur during initialize_async when _get_atomic_attacks_async resolves seed groups
        with pytest.raises(ValueError, match="DatasetConfiguration has no seed_groups"):
            await scenario.initialize_async(objective_target=mock_objective_target)


@pytest.mark.usefixtures(*FIXTURES)
class TestJailbreakAttackGeneration:
    """Tests for Jailbreak attack generation."""

    @pytest.mark.asyncio
    async def test_attack_generation_for_all(
        self, mock_objective_target, mock_objective_scorer, mock_memory_seed_groups
    ):
        """Test that _get_atomic_attacks_async returns atomic attacks."""
        with patch.object(Jailbreak, "_resolve_seed_groups", return_value=mock_memory_seed_groups):
            scenario = Jailbreak(objective_scorer=mock_objective_scorer)

            await scenario.initialize_async(objective_target=mock_objective_target)
            atomic_attacks = await scenario._get_atomic_attacks_async()

            assert len(atomic_attacks) > 0
            assert all(hasattr(run, "_attack") for run in atomic_attacks)

    @pytest.mark.asyncio
    async def test_attack_generation_for_pyrit(
        self, mock_objective_target, mock_objective_scorer, mock_memory_seed_groups, pyrit_jailbreak_strategy
    ):
        """Test that the single turn attack generation works."""
        with patch.object(Jailbreak, "_resolve_seed_groups", return_value=mock_memory_seed_groups):
            scenario = Jailbreak(
                objective_scorer=mock_objective_scorer,
            )

            await scenario.initialize_async(
                objective_target=mock_objective_target, scenario_strategies=[
                    pyrit_jailbreak_strategy]
            )
            atomic_attacks = await scenario._get_atomic_attacks_async()
            for run in atomic_attacks:
                assert isinstance(run._attack, PromptSendingAttack)

    @pytest.mark.asyncio
    async def test_attack_runs_include_objectives(
        self, mock_objective_target, mock_objective_scorer, mock_memory_seed_groups
    ):
        """Test that attack runs include objectives for each seed prompt."""
        with patch.object(Jailbreak, "_resolve_seed_groups", return_value=mock_memory_seed_groups):
            scenario = Jailbreak(
                objective_scorer=mock_objective_scorer,
            )

            await scenario.initialize_async(objective_target=mock_objective_target)
            atomic_attacks = await scenario._get_atomic_attacks_async()

            # Check that objectives are created for each seed prompt
            for run in atomic_attacks:
                assert len(run.objectives) > 0

    @pytest.mark.asyncio
    async def test_get_atomic_attacks_async_returns_attacks(
        self, mock_objective_target, mock_objective_scorer, mock_memory_seed_groups
    ):
        """Test that _get_atomic_attacks_async returns atomic attacks."""
        with patch.object(Jailbreak, "_resolve_seed_groups", return_value=mock_memory_seed_groups):
            scenario = Jailbreak(
                objective_scorer=mock_objective_scorer,
            )

            await scenario.initialize_async(objective_target=mock_objective_target)
            atomic_attacks = await scenario._get_atomic_attacks_async()
            assert len(atomic_attacks) > 0
            assert all(hasattr(run, "_attack") for run in atomic_attacks)


@pytest.mark.usefixtures(*FIXTURES)
class TestJailbreakLifecycle:
    """Tests for Jailbreak lifecycle."""

    @pytest.mark.asyncio
    async def test_initialize_async_with_max_concurrency(
        self,
        *,
        mock_objective_target: PromptTarget,
        mock_objective_scorer: TrueFalseInverterScorer,
        mock_memory_seed_groups: List[SeedGroup],
    ) -> None:
        """Test initialization with custom max_concurrency."""
        with patch.object(Jailbreak, "_resolve_seed_groups", return_value=mock_memory_seed_groups):
            scenario = Jailbreak(objective_scorer=mock_objective_scorer)
            await scenario.initialize_async(objective_target=mock_objective_target, max_concurrency=20)
            assert scenario._max_concurrency == 20

    @pytest.mark.asyncio
    async def test_initialize_async_with_memory_labels(
        self,
        *,
        mock_objective_target: PromptTarget,
        mock_objective_scorer: TrueFalseInverterScorer,
        mock_memory_seed_groups: List[SeedGroup],
    ) -> None:
        """Test initialization with memory labels."""
        memory_labels = {"type": "jailbreak", "category": "scenario"}

        with patch.object(Jailbreak, "_resolve_seed_groups", return_value=mock_memory_seed_groups):
            scenario = Jailbreak(objective_scorer=mock_objective_scorer)
            await scenario.initialize_async(
                memory_labels=memory_labels,
                objective_target=mock_objective_target,
            )
            assert scenario._memory_labels == memory_labels


@pytest.mark.usefixtures(*FIXTURES)
class TestJailbreakProperties:
    """Tests for Jailbreak properties."""

    def test_scenario_version_is_set(
        self,
        *,
        mock_objective_scorer: TrueFalseInverterScorer,
    ) -> None:
        """Test that scenario version is properly set."""
        scenario = Jailbreak(
            objective_scorer=mock_objective_scorer,
        )

        assert scenario.version == 1

    @pytest.mark.asyncio
    async def test_no_target_duplication_async(
        self, *, mock_objective_target: PromptTarget, mock_memory_seed_groups: List[SeedGroup]
    ) -> None:
        """Test that all three targets (adversarial, object, scorer) are distinct."""
        # TODO confirm that DatasetConfig does/doesn't intercept this
        with patch.object(Jailbreak, "_resolve_seed_groups", return_value=mock_memory_seed_groups):
            scenario = Jailbreak()
            await scenario.initialize_async(objective_target=mock_objective_target)

            objective_target = scenario._objective_target
            scorer_target = scenario._scorer_config.objective_scorer  # type: ignore

            assert objective_target != scorer_target
