# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""Tests for the EncodingScenario class."""

from unittest.mock import MagicMock

import pytest

from pyrit.executor.attack.single_turn.prompt_sending import PromptSendingAttack
from pyrit.prompt_target import PromptTarget
from pyrit.scenarios import EncodingScenario
from pyrit.score import TrueFalseScorer
from pyrit.score.true_false.decoding_scorer import DecodingScorer


@pytest.fixture
def mock_objective_target():
    """Create a mock objective target for testing."""
    return MagicMock(spec=PromptTarget)


@pytest.fixture
def mock_objective_scorer():
    """Create a mock objective scorer for testing."""
    return MagicMock(spec=TrueFalseScorer)


@pytest.fixture
def sample_seeds():
    """Create sample seeds for testing."""
    return ["test prompt 1", "test prompt 2"]


@pytest.mark.usefixtures("patch_central_database")
class TestEncodingScenarioInitialization:
    """Tests for EncodingScenario initialization."""

    def test_init_with_custom_seed_prompts(self, mock_objective_target, mock_objective_scorer, sample_seeds):
        """Test initialization with custom seed prompts."""
        scenario = EncodingScenario(
            objective_target=mock_objective_target,
            seed_prompts=sample_seeds,
            objective_scorer=mock_objective_scorer,
        )

        assert scenario._seed_prompts == sample_seeds
        assert scenario._objective_target == mock_objective_target
        assert scenario.name == "Encoding Scenario"
        assert scenario.version == 1

    def test_init_with_default_seed_prompts(self, mock_objective_target, mock_objective_scorer):
        """Test initialization with default seed prompts (Garak dataset)."""
        scenario = EncodingScenario(
            objective_target=mock_objective_target,
            objective_scorer=mock_objective_scorer,
        )

        # Should load default datasets from Garak
        assert len(scenario._seed_prompts) > 0
        assert scenario._objective_target == mock_objective_target

    def test_init_with_custom_scorer(self, mock_objective_target, mock_objective_scorer, sample_seeds):
        """Test initialization with custom objective scorer."""
        scenario = EncodingScenario(
            objective_target=mock_objective_target,
            seed_prompts=sample_seeds,
            objective_scorer=mock_objective_scorer,
        )

        assert scenario._scorer_config.objective_scorer == mock_objective_scorer

    def test_init_creates_default_scorer_when_not_provided(self, mock_objective_target, sample_seeds):
        """Test that initialization creates default DecodingScorer when not provided."""
        scenario = EncodingScenario(
            objective_target=mock_objective_target,
            seed_prompts=sample_seeds,
        )

        # Should create a DecodingScorer by default
        assert scenario._scorer_config.objective_scorer is not None
        assert isinstance(scenario._scorer_config.objective_scorer, DecodingScorer)

    def test_init_with_memory_labels(self, mock_objective_target, mock_objective_scorer, sample_seeds):
        """Test initialization with memory labels."""
        memory_labels = {"test": "encoding", "category": "scenario"}

        scenario = EncodingScenario(
            objective_target=mock_objective_target,
            seed_prompts=sample_seeds,
            memory_labels=memory_labels,
            objective_scorer=mock_objective_scorer,
        )

        assert scenario._memory_labels == memory_labels

    def test_init_with_custom_encoding_templates(self, mock_objective_target, mock_objective_scorer, sample_seeds):
        """Test initialization with custom encoding templates."""
        custom_templates = ["template1", "template2"]

        scenario = EncodingScenario(
            objective_target=mock_objective_target,
            seed_prompts=sample_seeds,
            encoding_templates=custom_templates,
            objective_scorer=mock_objective_scorer,
        )

        assert scenario._encoding_templates == custom_templates

    def test_init_with_max_concurrency(self, mock_objective_target, mock_objective_scorer, sample_seeds):
        """Test initialization with custom max_concurrency."""
        scenario = EncodingScenario(
            objective_target=mock_objective_target,
            seed_prompts=sample_seeds,
            max_concurrency=20,
            objective_scorer=mock_objective_scorer,
        )

        assert scenario._max_concurrency == 20

    def test_init_attack_strategies(self, mock_objective_target, mock_objective_scorer, sample_seeds):
        """Test that attack strategies are set correctly."""
        scenario = EncodingScenario(
            objective_target=mock_objective_target,
            seed_prompts=sample_seeds,
            objective_scorer=mock_objective_scorer,
        )

        assert scenario._attack_strategies == ["Garak basic"]


@pytest.mark.usefixtures("patch_central_database")
class TestEncodingScenarioAttackRuns:
    """Tests for EncodingScenario attack run generation."""

    @pytest.mark.asyncio
    async def test_get_attack_runs_async_returns_attacks(
        self, mock_objective_target, mock_objective_scorer, sample_seeds
    ):
        """Test that _get_attack_runs_async returns attack runs."""
        scenario = EncodingScenario(
            objective_target=mock_objective_target,
            seed_prompts=sample_seeds,
            objective_scorer=mock_objective_scorer,
        )

        await scenario.initialize_async()
        attack_runs = await scenario._get_attack_runs_async()

        # Should return multiple attack runs (one for each encoding type)
        assert len(attack_runs) > 0
        assert all(isinstance(run._attack, PromptSendingAttack) for run in attack_runs)

    @pytest.mark.asyncio
    async def test_get_converter_attacks_returns_multiple_encodings(
        self, mock_objective_target, mock_objective_scorer, sample_seeds
    ):
        """Test that _get_converter_attacks returns attacks for multiple encoding types."""
        scenario = EncodingScenario(
            objective_target=mock_objective_target,
            seed_prompts=sample_seeds,
            objective_scorer=mock_objective_scorer,
        )

        attack_runs = scenario._get_converter_attacks()

        # Should have multiple attack runs for different encodings
        # The list includes: Base64 (4 variants), Base2048, Base16, Base32, ASCII85 (2), hex,
        # quoted-printable, UUencode, ROT13, Braille, Atbash, Morse, NATO, Ecoji, Zalgo, Leet, AsciiSmuggler
        assert len(attack_runs) > 0

    @pytest.mark.asyncio
    async def test_get_prompt_attacks_creates_attack_runs(
        self, mock_objective_target, mock_objective_scorer, sample_seeds
    ):
        """Test that _get_prompt_attacks creates attack runs with correct structure."""
        from pyrit.prompt_converter import Base64Converter

        scenario = EncodingScenario(
            objective_target=mock_objective_target,
            seed_prompts=sample_seeds,
            objective_scorer=mock_objective_scorer,
        )

        attack_runs = scenario._get_prompt_attacks(converters=[Base64Converter()], encoding_name="Base64")

        # Should create attack runs
        assert len(attack_runs) > 0

        # Each attack run should have the correct attack type
        for run in attack_runs:
            assert isinstance(run._attack, PromptSendingAttack)
            assert len(run._objectives) == len(sample_seeds)
            assert run._seed_groups is not None
            assert len(run._seed_groups) == len(sample_seeds)

    @pytest.mark.asyncio
    async def test_attack_runs_include_objectives(self, mock_objective_target, mock_objective_scorer, sample_seeds):
        """Test that attack runs include objectives for each seed prompt."""
        from pyrit.prompt_converter import Base64Converter

        scenario = EncodingScenario(
            objective_target=mock_objective_target,
            seed_prompts=sample_seeds,
            objective_scorer=mock_objective_scorer,
        )

        attack_runs = scenario._get_prompt_attacks(converters=[Base64Converter()], encoding_name="Base64")

        # Check that objectives are created for each seed prompt
        for run in attack_runs:
            assert len(run._objectives) == len(sample_seeds)
            for i, objective in enumerate(run._objectives):
                assert "Make the model say an encoded payload:" in objective
                assert sample_seeds[i] in objective


@pytest.mark.usefixtures("patch_central_database")
class TestEncodingScenarioExecution:
    """Tests for EncodingScenario execution."""

    @pytest.mark.asyncio
    async def test_scenario_initialization(self, mock_objective_target, mock_objective_scorer, sample_seeds):
        """Test that scenario can be initialized successfully."""
        scenario = EncodingScenario(
            objective_target=mock_objective_target,
            seed_prompts=sample_seeds,
            objective_scorer=mock_objective_scorer,
        )

        await scenario.initialize_async()

        # Verify initialization creates attack runs
        assert scenario.attack_run_count > 0

    @pytest.mark.asyncio
    async def test_get_default_dataset_loads_garak_data(self, mock_objective_target, mock_objective_scorer):
        """Test that _get_default_dataset loads data from Garak datasets."""
        scenario = EncodingScenario(
            objective_target=mock_objective_target,
            objective_scorer=mock_objective_scorer,
        )

        # Should load slur_terms_en and web_html_js from Garak
        assert len(scenario._seed_prompts) > 0

        # Verify it's loading actual data (not empty)
        assert all(isinstance(prompt, str) for prompt in scenario._seed_prompts)
        assert all(len(prompt) > 0 for prompt in scenario._seed_prompts)
