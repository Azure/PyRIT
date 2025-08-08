# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from pyrit.exceptions import InvalidJsonException
from pyrit.memory.central_memory import CentralMemory
from pyrit.models import PromptRequestPiece, PromptRequestResponse, Score
from pyrit.prompt_target import PromptChatTarget
from pyrit.score.policy_scorer import PolicyScorer


@pytest.fixture
def valid_scale_args():
    """Fixture providing valid scale arguments for PolicyScorer initialization."""
    return {
        "minimum_value": 1,
        "maximum_value": 5,
        "category": "test_category",
        "task": "Test task description for policy generation",
    }


@pytest.fixture
def mock_policy_generator():
    """Fixture providing a mock policy generator target."""
    return MagicMock(spec=PromptChatTarget)


@pytest.fixture
def mock_evaluator():
    """Fixture providing a mock evaluator target."""
    return MagicMock(spec=PromptChatTarget)


@pytest.fixture
def policy_generation_response():
    """Fixture providing a mock policy generation response."""
    json_response = {
        "policy": [
            "The response should be accurate and factual",
            "The response should be helpful to the user",
            "The response should not contain harmful content",
        ]
    }

    return PromptRequestResponse(
        request_pieces=[PromptRequestPiece(role="assistant", original_value=json.dumps(json_response))]
    )


@pytest.fixture
def scoring_response():
    """Fixture providing a mock scoring response."""
    json_response = {
        "score": [4, 3, 5],
        "descriptions": [
            "Response is factually accurate",
            "Response is somewhat helpful",
            "Response contains no harmful content",
        ],
        "rationales": [
            "The information provided appears to be correct",
            "The response addresses the question but could be more detailed",
            "No harmful, toxic, or inappropriate content detected",
        ],
    }

    return PromptRequestResponse(
        request_pieces=[PromptRequestPiece(role="assistant", original_value=json.dumps(json_response))]
    )


@pytest.fixture
def mock_request_piece(patch_central_database):
    """Fixture providing a mock request piece to be scored."""
    memory = CentralMemory.get_memory_instance()
    request = PromptRequestPiece(
        role="user",
        original_value="This is a test response to be scored",
        original_value_data_type="text",
        conversation_id="test-conv",
        sequence=1,
    )
    memory.add_request_pieces_to_memory(request_pieces=[request])
    return request


class TestPolicyScorerInitialization:
    """Test cases for PolicyScorer initialization."""

    def test_policy_scorer_init_with_valid_args(
        self, mock_policy_generator, mock_evaluator, valid_scale_args, patch_central_database
    ):
        """Test successful initialization with valid arguments."""
        scorer = PolicyScorer(
            policy_generator=mock_policy_generator,
            evaluator=mock_evaluator,
            scale_args=valid_scale_args,
        )

        assert scorer._policy_generator == mock_policy_generator
        assert scorer._evaluator == mock_evaluator
        assert scorer.scorer_type == "float_scale"
        assert scorer._minimum_value == 1
        assert scorer._maximum_value == 5
        assert scorer._category == "test_category"
        assert scorer._task == "Test task description for policy generation"
        assert scorer.policies is None

    def test_policy_scorer_init_with_custom_prompt_paths(
        self, mock_policy_generator, mock_evaluator, valid_scale_args, patch_central_database
    ):
        """Test initialization with custom prompt paths."""
        custom_policy_path = Path("/custom/policy/path.yaml")
        custom_score_path = Path("/custom/score/path.yaml")

        with patch("pyrit.models.SeedPrompt.from_yaml_file") as mock_seed_prompt:
            mock_template = MagicMock()
            mock_template.render_template_value_silent.return_value = "rendered prompt"
            mock_seed_prompt.return_value = mock_template

            scorer = PolicyScorer(
                policy_generator=mock_policy_generator,
                evaluator=mock_evaluator,
                scale_args=valid_scale_args,
                policy_generation_prompt_path=custom_policy_path,
                score_prompt_path=custom_score_path,
            )

            assert scorer is not None
            # Verify that custom paths were used
            assert mock_seed_prompt.call_count == 2

    def test_policy_scorer_init_missing_required_scale_args(
        self, mock_policy_generator, mock_evaluator, patch_central_database
    ):
        """Test initialization fails with missing required scale arguments."""
        incomplete_args = {
            "minimum_value": 1,
            "maximum_value": 5,
            # Missing 'category' and 'task'
        }

        with pytest.raises(ValueError, match="Missing key in scale_args"):
            PolicyScorer(
                policy_generator=mock_policy_generator,
                evaluator=mock_evaluator,
                scale_args=incomplete_args,
            )

    def test_policy_scorer_init_invalid_scale_args_types(
        self, mock_policy_generator, mock_evaluator, patch_central_database
    ):
        """Test initialization fails with invalid scale argument types."""
        invalid_args = {
            "minimum_value": "1",  # Should be int
            "maximum_value": 5,
            "category": "test_category",
            "task": "Test task",
        }

        with pytest.raises(ValueError, match="Minimum value must be an integer"):
            PolicyScorer(
                policy_generator=mock_policy_generator,
                evaluator=mock_evaluator,
                scale_args=invalid_args,
            )

    def test_policy_scorer_init_invalid_min_max_values(
        self, mock_policy_generator, mock_evaluator, patch_central_database
    ):
        """Test initialization fails when minimum value is greater than maximum."""
        invalid_args = {
            "minimum_value": 5,
            "maximum_value": 1,  # Invalid: min > max
            "category": "test_category",
            "task": "Test task",
        }

        with pytest.raises(ValueError, match="Minimum value must be less than or equal to the maximum value"):
            PolicyScorer(
                policy_generator=mock_policy_generator,
                evaluator=mock_evaluator,
                scale_args=invalid_args,
            )

    def test_policy_scorer_init_empty_category_or_task(
        self, mock_policy_generator, mock_evaluator, patch_central_database
    ):
        """Test initialization fails with empty category or task."""
        invalid_args = {
            "minimum_value": 1,
            "maximum_value": 5,
            "category": "",  # Empty category
            "task": "Test task",
        }

        with pytest.raises(ValueError, match="Category must be set and cannot be empty"):
            PolicyScorer(
                policy_generator=mock_policy_generator,
                evaluator=mock_evaluator,
                scale_args=invalid_args,
            )


class TestPolicyGeneration:
    """Test cases for policy generation functionality."""

    @pytest.mark.asyncio
    async def test_generate_policies_success(
        self,
        mock_policy_generator,
        mock_evaluator,
        valid_scale_args,
        policy_generation_response,
        patch_central_database,
    ):
        """Test successful policy generation."""
        mock_policy_generator.send_prompt_async = AsyncMock(return_value=policy_generation_response)

        scorer = PolicyScorer(
            policy_generator=mock_policy_generator,
            evaluator=mock_evaluator,
            scale_args=valid_scale_args,
        )

        policies = await scorer._generate_policies()

        assert len(policies) == 3
        assert "The response should be accurate and factual" in policies
        assert "The response should be helpful to the user" in policies
        assert "The response should not contain harmful content" in policies

        # Verify the policy generator was called
        mock_policy_generator.send_prompt_async.assert_called_once()

    @pytest.mark.asyncio
    async def test_generate_policies_communication_error(
        self, mock_policy_generator, mock_evaluator, valid_scale_args, patch_central_database
    ):
        """Test policy generation with communication error."""
        mock_policy_generator.send_prompt_async = AsyncMock(side_effect=Exception("Connection error"))

        scorer = PolicyScorer(
            policy_generator=mock_policy_generator,
            evaluator=mock_evaluator,
            scale_args=valid_scale_args,
        )

        with pytest.raises(Exception, match="Error policy prompt"):
            await scorer._generate_policies()

    @pytest.mark.asyncio
    async def test_generate_policies_invalid_json_response(
        self, mock_policy_generator, mock_evaluator, valid_scale_args, patch_central_database
    ):
        """Test policy generation with invalid JSON response."""
        invalid_response = PromptRequestResponse(
            request_pieces=[PromptRequestPiece(role="assistant", original_value="invalid json")]
        )
        mock_policy_generator.send_prompt_async = AsyncMock(return_value=invalid_response)

        scorer = PolicyScorer(
            policy_generator=mock_policy_generator,
            evaluator=mock_evaluator,
            scale_args=valid_scale_args,
        )

        with pytest.raises(InvalidJsonException, match="Invalid JSON response"):
            await scorer._generate_policies()

    @pytest.mark.asyncio
    async def test_generate_policies_missing_policy_key(
        self, mock_policy_generator, mock_evaluator, valid_scale_args, patch_central_database
    ):
        """Test policy generation with response missing 'policy' key."""
        invalid_response = PromptRequestResponse(
            request_pieces=[
                PromptRequestPiece(role="assistant", original_value=json.dumps({"wrong_key": ["policy1", "policy2"]}))
            ]
        )
        mock_policy_generator.send_prompt_async = AsyncMock(return_value=invalid_response)

        scorer = PolicyScorer(
            policy_generator=mock_policy_generator,
            evaluator=mock_evaluator,
            scale_args=valid_scale_args,
        )

        with pytest.raises(InvalidJsonException, match="Invalid JSON response"):
            await scorer._generate_policies()


class TestWeightAdjustment:
    """Test cases for weight adjustment functionality."""

    def test_validate_human_weight_valid_values(
        self, mock_policy_generator, mock_evaluator, valid_scale_args, patch_central_database
    ):
        """Test weight validation with valid values."""
        scorer = PolicyScorer(
            policy_generator=mock_policy_generator,
            evaluator=mock_evaluator,
            scale_args=valid_scale_args,
        )

        # Test valid weight values
        assert scorer._validate_human_weight("0.5") == 0.5
        assert scorer._validate_human_weight("0.0") == 0.0
        assert scorer._validate_human_weight("1.0") == 1.0
        assert scorer._validate_human_weight("0.75") == 0.75

    def test_validate_human_weight_invalid_values(
        self, mock_policy_generator, mock_evaluator, valid_scale_args, patch_central_database
    ):

        # Test invalid numeric values
        with pytest.raises(ValueError, match="Weights require a numeric value. Got -0.1"):
            """Test weight validation with invalid values."""
            scorer = PolicyScorer(
                policy_generator=mock_policy_generator,
                evaluator=mock_evaluator,
                scale_args=valid_scale_args,
            )
            scorer._validate_human_weight("-0.1")

        with pytest.raises(ValueError, match="Weights require a numeric value. Got 1.1"):
            scorer = PolicyScorer(
                policy_generator=mock_policy_generator,
                evaluator=mock_evaluator,
                scale_args=valid_scale_args,
            )
            scorer._validate_human_weight("1.1")

        # Test non-numeric values
        with pytest.raises(ValueError, match="Weights require a numeric value. Got invalid"):
            scorer = PolicyScorer(
                policy_generator=mock_policy_generator,
                evaluator=mock_evaluator,
                scale_args=valid_scale_args,
            )
            scorer._validate_human_weight("invalid")

    @patch("tkinter.Tk")
    @patch("tkinter.simpledialog.askstring")
    def test_get_user_input_success(
        self, mock_askstring, mock_tk, mock_policy_generator, mock_evaluator, valid_scale_args, patch_central_database
    ):
        """Test successful user input collection."""
        mock_root = MagicMock()
        mock_tk.return_value = mock_root
        mock_askstring.return_value = "  0.5  "  # Test with whitespace

        scorer = PolicyScorer(
            policy_generator=mock_policy_generator,
            evaluator=mock_evaluator,
            scale_args=valid_scale_args,
        )

        result = scorer._get_user_input("Test message")

        assert result == "0.5"
        mock_root.withdraw.assert_called_once()
        mock_root.destroy.assert_called_once()
        mock_askstring.assert_called_once_with("Score Prompt", "Test message")

    @patch("tkinter.Tk")
    @patch("tkinter.simpledialog.askstring")
    def test_get_user_input_cancelled(
        self, mock_askstring, mock_tk, mock_policy_generator, mock_evaluator, valid_scale_args, patch_central_database
    ):
        """Test user input when dialog is cancelled."""
        mock_root = MagicMock()
        mock_tk.return_value = mock_root
        mock_askstring.return_value = None  # User cancelled

        scorer = PolicyScorer(
            policy_generator=mock_policy_generator,
            evaluator=mock_evaluator,
            scale_args=valid_scale_args,
        )

        result = scorer._get_user_input("Test message")

        assert result == ""

    def test_get_user_input_no_tkinter(
        self, mock_policy_generator, mock_evaluator, valid_scale_args, patch_central_database
    ):
        """Test user input when tkinter is not available."""
        scorer = PolicyScorer(
            policy_generator=mock_policy_generator,
            evaluator=mock_evaluator,
            scale_args=valid_scale_args,
        )

        with patch("builtins.__import__", side_effect=ImportError("No module named 'tkinter'")):
            with pytest.raises(ImportError):
                scorer._get_user_input("Test message")

    @patch.object(PolicyScorer, "_get_user_input")
    def test_adjust_weight_manually_success(
        self, mock_get_input, mock_policy_generator, mock_evaluator, valid_scale_args, patch_central_database
    ):
        """Test successful manual weight adjustment."""
        mock_get_input.side_effect = ["0.5", "0.3", "0.2"]

        scorer = PolicyScorer(
            policy_generator=mock_policy_generator,
            evaluator=mock_evaluator,
            scale_args=valid_scale_args,
        )
        scorer.policies = ["Policy 1", "Policy 2", "Policy 3"]

        weights = scorer._adjust_weight_manually()

        assert weights == [0.5, 0.3, 0.2]
        assert mock_get_input.call_count == 3

    @patch.object(PolicyScorer, "_get_user_input")
    def test_adjust_weight_manually_retry_on_invalid(
        self, mock_get_input, mock_policy_generator, mock_evaluator, valid_scale_args, patch_central_database
    ):
        """Test weight adjustment with retry on invalid input."""
        # First attempt invalid, second attempt valid
        mock_get_input.side_effect = ["invalid", "0.5"]

        scorer = PolicyScorer(
            policy_generator=mock_policy_generator,
            evaluator=mock_evaluator,
            scale_args=valid_scale_args,
        )
        scorer.policies = ["Policy 1"]

        with patch("builtins.print"):  # Suppress print output
            weights = scorer._adjust_weight_manually()

        assert weights == [0.5]
        assert mock_get_input.call_count == 2


class TestScoring:
    """Test cases for scoring functionality."""

    @pytest.mark.asyncio
    async def test_score_dimension_value_with_llm_success(
        self, mock_policy_generator, mock_evaluator, valid_scale_args, scoring_response, patch_central_database
    ):
        """Test successful scoring with LLM."""
        mock_evaluator.send_prompt_async = AsyncMock(return_value=scoring_response)
        mock_evaluator.set_system_prompt = MagicMock()
        mock_evaluator.get_identifier = MagicMock(return_value={"name": "test_evaluator"})

        scorer = PolicyScorer(
            policy_generator=mock_policy_generator,
            evaluator=mock_evaluator,
            scale_args=valid_scale_args,
        )
        scorer.policies = ["Policy 1", "Policy 2", "Policy 3"]

        unvalidated_scores = await scorer._score_dimension_value_with_llm(
            prompt_target=mock_evaluator,
            system_prompt="System prompt",
            prompt_request_value="Test content",
            prompt_request_data_type="text",
            scored_prompt_id="test-id",
            category="test_category",
            task="test_task",
        )

        assert len(unvalidated_scores) == 3
        assert unvalidated_scores[0].raw_score_value == "4"
        assert unvalidated_scores[1].raw_score_value == "3"
        assert unvalidated_scores[2].raw_score_value == "5"

        # Verify system prompt was set
        mock_evaluator.set_system_prompt.assert_called_once()
        mock_evaluator.send_prompt_async.assert_called_once()

    @pytest.mark.asyncio
    async def test_score_dimension_value_with_llm_communication_error(
        self, mock_policy_generator, mock_evaluator, valid_scale_args, patch_central_database
    ):
        """Test scoring with communication error."""
        mock_evaluator.send_prompt_async = AsyncMock(side_effect=Exception("Connection error"))
        mock_evaluator.set_system_prompt = MagicMock()

        scorer = PolicyScorer(
            policy_generator=mock_policy_generator,
            evaluator=mock_evaluator,
            scale_args=valid_scale_args,
        )
        scorer.policies = ["Policy 1"]

        with pytest.raises(Exception, match="Error scoring prompt"):
            await scorer._score_dimension_value_with_llm(
                prompt_target=mock_evaluator,
                system_prompt="System prompt",
                prompt_request_value="Test content",
                prompt_request_data_type="text",
                scored_prompt_id="test-id",
            )

    @pytest.mark.asyncio
    async def test_score_dimension_value_with_llm_invalid_json(
        self, mock_policy_generator, mock_evaluator, valid_scale_args, patch_central_database
    ):
        """Test scoring with invalid JSON response."""
        invalid_response = PromptRequestResponse(
            request_pieces=[PromptRequestPiece(role="assistant", original_value="invalid json")]
        )
        mock_evaluator.send_prompt_async = AsyncMock(return_value=invalid_response)
        mock_evaluator.set_system_prompt = MagicMock()

        scorer = PolicyScorer(
            policy_generator=mock_policy_generator,
            evaluator=mock_evaluator,
            scale_args=valid_scale_args,
        )
        scorer.policies = ["Policy 1"]

        with pytest.raises(InvalidJsonException, match="Invalid or malformed JSON response"):
            await scorer._score_dimension_value_with_llm(
                prompt_target=mock_evaluator,
                system_prompt="System prompt",
                prompt_request_value="Test content",
                prompt_request_data_type="text",
                scored_prompt_id="test-id",
            )

    @pytest.mark.asyncio
    async def test_score_async_full_workflow(
        self,
        mock_policy_generator,
        mock_evaluator,
        valid_scale_args,
        policy_generation_response,
        scoring_response,
        mock_request_piece,
        patch_central_database,
    ):
        """Test the complete scoring workflow."""
        # Mock policy generation
        mock_policy_generator.send_prompt_async = AsyncMock(return_value=policy_generation_response)

        # Mock scoring
        mock_evaluator.send_prompt_async = AsyncMock(return_value=scoring_response)
        mock_evaluator.set_system_prompt = MagicMock()
        mock_evaluator.get_identifier = MagicMock(return_value={"name": "test_evaluator"})

        scorer = PolicyScorer(
            policy_generator=mock_policy_generator,
            evaluator=mock_evaluator,
            scale_args=valid_scale_args,
        )

        # Mock weight adjustment
        with patch.object(scorer, "_adjust_weight_manually", return_value=[0.5, 0.3, 0.2]):
            scores = await scorer._score_async(mock_request_piece, task="test_task")

        assert len(scores) == 1
        assert isinstance(scores[0], Score)
        assert scores[0].score_category == "test_category"
        assert scores[0].score_type == "float_scale"
        assert scores[0].task == "test_task"

        # Verify policies were generated and weights assigned
        assert scorer.policies is not None
        assert len(scorer.policies) == 3
        assert scorer.weights == [0.5, 0.3, 0.2]

    @pytest.mark.asyncio
    async def test_score_async_reuse_policies_and_weights(
        self,
        mock_policy_generator,
        mock_evaluator,
        valid_scale_args,
        scoring_response,
        mock_request_piece,
        patch_central_database,
    ):
        """Test that policies and weights are reused on subsequent scoring calls."""
        # Mock scoring only (no policy generation needed on second call)
        mock_evaluator.send_prompt_async = AsyncMock(return_value=scoring_response)
        mock_evaluator.set_system_prompt = MagicMock()
        mock_evaluator.get_identifier = MagicMock(return_value={"name": "test_evaluator"})

        scorer = PolicyScorer(
            policy_generator=mock_policy_generator,
            evaluator=mock_evaluator,
            scale_args=valid_scale_args,
        )

        # Set pre-existing policies and weights
        scorer.policies = ["Policy 1", "Policy 2", "Policy 3"]
        scorer.weights = [0.4, 0.4, 0.2]

        scores = await scorer._score_async(mock_request_piece, task="test_task")

        assert len(scores) == 1

        # Verify policy generator was not called (policies already exist)
        mock_policy_generator.send_prompt_async.assert_not_called()

        # Verify evaluator was called
        mock_evaluator.send_prompt_async.assert_called_once()


class TestValidation:
    """Test cases for validation functionality."""

    def test_validate_success(
        self, mock_policy_generator, mock_evaluator, valid_scale_args, mock_request_piece, patch_central_database
    ):
        """Test successful validation."""
        scorer = PolicyScorer(
            policy_generator=mock_policy_generator,
            evaluator=mock_evaluator,
            scale_args=valid_scale_args,
        )

        # Should not raise any exception
        scorer.validate(mock_request_piece, task="test_task")

    def test_validate_invalid_data_type(
        self, mock_policy_generator, mock_evaluator, valid_scale_args, patch_central_database
    ):
        """Test validation with invalid data type."""
        invalid_request = PromptRequestPiece(
            role="user",
            original_value="test",
            converted_value="test",
            original_value_data_type="image_path",  # Invalid: should be "text"
        )

        scorer = PolicyScorer(
            policy_generator=mock_policy_generator,
            evaluator=mock_evaluator,
            scale_args=valid_scale_args,
        )

        with pytest.raises(ValueError, match="The original value data type must be text"):
            scorer.validate(invalid_request, task="test_task")

    def test_validate_missing_task(
        self, mock_policy_generator, mock_evaluator, valid_scale_args, mock_request_piece, patch_central_database
    ):
        """Test validation with missing task."""
        scorer = PolicyScorer(
            policy_generator=mock_policy_generator,
            evaluator=mock_evaluator,
            scale_args=valid_scale_args,
        )

        with pytest.raises(ValueError, match="Task must be provided"):
            scorer.validate(mock_request_piece, task=None)

        with pytest.raises(ValueError, match="Task must be provided"):
            scorer.validate(mock_request_piece, task="")


class TestScaleArgumentsValidation:
    """Test cases for scale arguments validation."""

    def test_validate_scale_arguments_all_valid(self, mock_policy_generator, mock_evaluator, patch_central_database):
        """Test validation with all valid scale arguments."""
        valid_args = {
            "minimum_value": 1,
            "maximum_value": 10,
            "category": "test_category",
            "task": "Test task description",
        }

        # Should not raise any exception
        scorer = PolicyScorer(
            policy_generator=mock_policy_generator,
            evaluator=mock_evaluator,
            scale_args=valid_args,
        )
        assert scorer is not None

    def test_validate_scale_arguments_missing_keys(self, mock_policy_generator, mock_evaluator, patch_central_database):
        """Test validation with missing required keys."""
        test_cases = [
            ({}, "minimum_value"),
            ({"minimum_value": 1}, "maximum_value"),
            ({"minimum_value": 1, "maximum_value": 5}, "category"),
            ({"minimum_value": 1, "maximum_value": 5, "category": "test"}, "task"),
        ]

        for incomplete_args, expected_missing_key in test_cases:
            with pytest.raises(ValueError, match=f"Missing key in scale_args: {expected_missing_key}"):
                PolicyScorer(
                    policy_generator=mock_policy_generator,
                    evaluator=mock_evaluator,
                    scale_args=incomplete_args,
                )

    def test_validate_scale_arguments_invalid_types(
        self, mock_policy_generator, mock_evaluator, patch_central_database
    ):
        """Test validation with invalid argument types."""
        base_args = {
            "minimum_value": 1,
            "maximum_value": 5,
            "category": "test_category",
            "task": "Test task",
        }

        test_cases = [
            ({"minimum_value": "1"}, "Minimum value must be an integer"),
            ({"maximum_value": "5"}, "Maximum value must be an integer"),
            ({"minimum_value": 1.5}, "Minimum value must be an integer"),
            ({"maximum_value": 5.5}, "Maximum value must be an integer"),
        ]

        for invalid_update, expected_error in test_cases:
            invalid_args = {**base_args, **invalid_update}
            with pytest.raises(ValueError, match=expected_error):
                PolicyScorer(
                    policy_generator=mock_policy_generator,
                    evaluator=mock_evaluator,
                    scale_args=invalid_args,
                )

    def test_validate_scale_arguments_invalid_range(
        self, mock_policy_generator, mock_evaluator, patch_central_database
    ):
        """Test validation with invalid min/max range."""
        invalid_args = {
            "minimum_value": 10,
            "maximum_value": 5,  # max < min
            "category": "test_category",
            "task": "Test task",
        }

        with pytest.raises(ValueError, match="Minimum value must be less than or equal to the maximum value"):
            PolicyScorer(
                policy_generator=mock_policy_generator,
                evaluator=mock_evaluator,
                scale_args=invalid_args,
            )

    def test_validate_scale_arguments_empty_strings(
        self, mock_policy_generator, mock_evaluator, patch_central_database
    ):
        """Test validation with empty string values."""
        base_args = {
            "minimum_value": 1,
            "maximum_value": 5,
            "category": "test_category",
            "task": "Test task",
        }

        test_cases = [
            ({"category": ""}, "Category must be set and cannot be empty"),
            ({"task": ""}, "Task must be set and cannot be empty"),
        ]

        for invalid_update, expected_error in test_cases:
            invalid_args = {**base_args, **invalid_update}
            with pytest.raises(ValueError, match=expected_error):
                PolicyScorer(
                    policy_generator=mock_policy_generator,
                    evaluator=mock_evaluator,
                    scale_args=invalid_args,
                )
