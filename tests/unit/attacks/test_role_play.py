# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import pathlib
import tempfile
import uuid
from unittest.mock import AsyncMock, MagicMock

import pytest
import yaml

from pyrit.attacks.base.attack_config import AttackConverterConfig, AttackScoringConfig
from pyrit.attacks.base.attack_context import SingleTurnAttackContext
from pyrit.attacks.single_turn.role_play import RolePlayAttack
from pyrit.models import AttackOutcome, PromptRequestResponse
from pyrit.models.attack_result import AttackResult
from pyrit.prompt_converter import Base64Converter, StringJoinConverter
from pyrit.prompt_normalizer import PromptConverterConfiguration
from pyrit.prompt_target import PromptChatTarget
from pyrit.score import Scorer


@pytest.fixture
def mock_objective_target():
    """Create a mock prompt target for testing"""
    target = MagicMock(spec=PromptChatTarget)
    target.send_prompt_async = AsyncMock()
    target.get_identifier.return_value = {"id": "mock_target_id"}
    return target


@pytest.fixture
def mock_adversarial_chat_target():
    """Create a mock adversarial chat target for testing"""
    target = MagicMock(spec=PromptChatTarget)
    target.send_prompt_async = AsyncMock()
    target.get_identifier.return_value = {"id": "mock_adversarial_chat_id"}
    return target


@pytest.fixture
def sample_response():
    """Create a sample response for testing"""
    return PromptRequestResponse.from_prompt(
        prompt="Test response content",
        role="assistant",
    )


@pytest.fixture
def mock_true_false_scorer():
    """Create a mock true/false scorer for testing"""
    scorer = MagicMock(spec=Scorer)
    scorer.scorer_type = "true_false"
    scorer.score_text_async = AsyncMock()
    return scorer


@pytest.fixture
def role_play_definition_file():
    """Create a temporary role play definition file for testing"""
    role_play_data = {
        "dataset_name": "test_role_play",
        "description": "Test role play definition",
        "authors": "Test Author",
        "groups": "Test Group",
        "prompts": [
            {
                "description": "This is used to rephrase the objective into a movie script format.",
                "parameters": ["objective"],
                "value": "DIRECTOR: Please rephrase the following objective into a script format: {{ objective }}",
            },
            {
                "description": "This is the 1st turn.",
                "parameters": "objective",
                "value": "Example text",
            },
            {
                "description": "This is the assistant turn.",
                "value": "Example text",
            },
        ],
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(role_play_data, f)
        temp_path = pathlib.Path(f.name)

    yield temp_path

    # Cleanup
    temp_path.unlink()


@pytest.fixture
def role_play_attack(mock_objective_target, mock_adversarial_chat_target, role_play_definition_file):
    """Create a RolePlayAttack instance with default configuration"""
    return RolePlayAttack(
        objective_target=mock_objective_target,
        adversarial_chat=mock_adversarial_chat_target,
        role_play_definition_path=role_play_definition_file,
    )


@pytest.fixture
def basic_context():
    """Create a basic context for testing"""
    return SingleTurnAttackContext(objective="test objective", conversation_id=str(uuid.uuid4()))


@pytest.mark.usefixtures("patch_central_database")
class TestRolePlayAttackInitialization:
    """Tests for RolePlayAttack initialization"""

    def test_init(self, mock_objective_target, mock_adversarial_chat_target, role_play_definition_file):
        """Test RolePlayAttack initialization with default parameters"""
        attack = RolePlayAttack(
            objective_target=mock_objective_target,
            adversarial_chat=mock_adversarial_chat_target,
            role_play_definition_path=role_play_definition_file,
        )

        assert attack._objective_target == mock_objective_target
        assert attack._adversarial_chat == mock_adversarial_chat_target
        assert attack._objective_scorer is None
        assert attack._max_attempts_on_failure == 0

    def test_init_with_valid_true_false_scorer(
        self, mock_objective_target, mock_adversarial_chat_target, role_play_definition_file, mock_scorer
    ):
        """Test RolePlayAttack initialization with a valid true/false scorer"""
        attack = RolePlayAttack(
            objective_target=mock_objective_target,
            adversarial_chat=mock_adversarial_chat_target,
            role_play_definition_path=role_play_definition_file,
            attack_scoring_config=AttackScoringConfig(objective_scorer=mock_scorer),
        )

        assert attack._objective_scorer == mock_scorer

    def test_init_with_invalid_scorer_type(
        self, mock_objective_target, mock_adversarial_chat_target, role_play_definition_file
    ):
        """Test RolePlayAttack initialization with an invalid scorer type"""
        scorer = MagicMock(spec=Scorer)
        scorer.scorer_type = "float_scale"
        with pytest.raises(ValueError, match="Objective scorer must be a true/false scorer"):
            RolePlayAttack(
                objective_target=mock_objective_target,
                adversarial_chat=mock_adversarial_chat_target,
                role_play_definition_path=role_play_definition_file,
                attack_scoring_config=AttackScoringConfig(objective_scorer=scorer),
            )

    def test_init_with_invalid_role_play_definition_path(self, mock_objective_target, mock_adversarial_chat_target):
        """Test RolePlayAttack initialization with an invalid role play definition path"""
        invalid_path = pathlib.Path("invalid/path/to/role_play_definition.yaml")
        with pytest.raises(FileNotFoundError):
            RolePlayAttack(
                objective_target=mock_objective_target,
                adversarial_chat=mock_adversarial_chat_target,
                role_play_definition_path=invalid_path,
            )

    def test_init_with_custom_parameters(
        self, mock_objective_target, mock_adversarial_chat_target, role_play_definition_file, mock_scorer
    ):
        """Test RolePlayAttack initialization with custom parameters"""
        request_converters = [PromptConverterConfiguration(converters=[Base64Converter()])]
        response_converters = [PromptConverterConfiguration(converters=[StringJoinConverter()])]

        attack = RolePlayAttack(
            objective_target=mock_objective_target,
            adversarial_chat=mock_adversarial_chat_target,
            role_play_definition_path=role_play_definition_file,
            attack_converter_config=AttackConverterConfig(
                request_converters=request_converters, response_converters=response_converters
            ),
            attack_scoring_config=AttackScoringConfig(objective_scorer=mock_scorer, auxiliary_scorers=[mock_scorer]),
            max_attempts_on_failure=3,
        )

        assert attack._objective_target == mock_objective_target
        assert attack._adversarial_chat == mock_adversarial_chat_target
        assert attack._max_attempts_on_failure == 3
        assert attack._objective_scorer == mock_scorer
        assert len(attack._auxiliary_scorers) == 1

    def test_init_with_negative_max_attempts(
        self, mock_objective_target, mock_adversarial_chat_target, role_play_definition_file
    ):
        """Test that initialization fails with negative max_attempts_on_failure"""
        with pytest.raises(ValueError, match="max_attempts_on_failure must be a non-negative integer"):
            RolePlayAttack(
                objective_target=mock_objective_target,
                adversarial_chat=mock_adversarial_chat_target,
                role_play_definition_path=role_play_definition_file,
                max_attempts_on_failure=-1,
            )

    def test_init_loads_role_play_definition_correctly(
        self, mock_objective_target, mock_adversarial_chat_target, role_play_definition_file
    ):
        """Test that role play definitions are loaded correctly from YAML"""
        attack = RolePlayAttack(
            objective_target=mock_objective_target,
            adversarial_chat=mock_adversarial_chat_target,
            role_play_definition_path=role_play_definition_file,
        )

        assert attack._rephrase_instructions is not None
        assert attack._user_start_turn is not None
        assert attack._assistant_start_turn is not None
        assert "{{ objective }}" in attack._rephrase_instructions.value

    def test_rephrase_converter_integration(
        self, mock_objective_target, mock_adversarial_chat_target, role_play_definition_file
    ):
        """Test that the rephrase converter is properly integrated into request converters"""
        attack = RolePlayAttack(
            objective_target=mock_objective_target,
            adversarial_chat=mock_adversarial_chat_target,
            role_play_definition_path=role_play_definition_file,
        )

        assert attack._request_converters is not None
        assert len(attack._request_converters) > 0
        converters = attack._request_converters[0].converters
        assert any("LLMGenericTextConverter" in str(type(converter)) for converter in converters)


@pytest.mark.usefixtures("patch_central_database")
class TestRolePlayAttack:
    """Tests for the RolePlayAttack attack method"""

    @pytest.mark.asyncio
    async def test_attack_simple(self, role_play_attack, basic_context):
        """Test a basic attack run"""
        role_play_attack._validate_context = MagicMock()
        role_play_attack._setup_async = AsyncMock()

        mock_result = AttackResult(
            conversation_id=basic_context.conversation_id,
            objective=basic_context.objective,
            attack_identifier=role_play_attack.get_identifier(),
            outcome=AttackOutcome.SUCCESS,
        )

        role_play_attack._perform_attack_async = AsyncMock(return_value=mock_result)

        result = await role_play_attack.execute_with_context_async(context=basic_context)

        # Verify result and proper execution order
        assert result == mock_result
        role_play_attack._validate_context.assert_called_once_with(context=basic_context)
        role_play_attack._setup_async.assert_called_once_with(context=basic_context)
        role_play_attack._perform_attack_async.assert_called_once_with(context=basic_context)
