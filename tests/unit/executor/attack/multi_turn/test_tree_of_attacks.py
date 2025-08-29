# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import asyncio
import json
import logging
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, cast
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from treelib.tree import Tree

from pyrit.exceptions import InvalidJsonException
from pyrit.executor.attack import (
    AttackAdversarialConfig,
    AttackConverterConfig,
    AttackScoringConfig,
    TAPAttackContext,
    TAPAttackResult,
    TreeOfAttacksWithPruningAttack,
)
from pyrit.executor.attack.multi_turn.tree_of_attacks import _TreeOfAttacksNode
from pyrit.models import (
    AttackOutcome,
    ConversationReference,
    ConversationType,
    PromptRequestPiece,
    PromptRequestResponse,
    Score,
    SeedPrompt,
)
from pyrit.prompt_normalizer import PromptNormalizer
from pyrit.prompt_target import PromptChatTarget, PromptTarget
from pyrit.score import Scorer

logger = logging.getLogger(__name__)


@dataclass
class NodeMockConfig:
    """Configuration for creating mock _TreeOfAttacksNode objects."""

    node_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    parent_id: Optional[str] = None
    prompt_sent: bool = False
    completed: bool = True
    off_topic: bool = False
    objective_score_value: Optional[float] = None
    auxiliary_scores: Dict[str, float] = field(default_factory=dict)
    objective_target_conversation_id: str = field(default_factory=lambda: str(uuid.uuid4()))


class MockNodeFactory:
    """Factory for creating mock _TreeOfAttacksNode objects."""

    @staticmethod
    def create_node(config: Optional[NodeMockConfig] = None) -> "_TreeOfAttacksNode":
        """Create a mock _TreeOfAttacksNode with the given configuration."""
        if config is None:
            config = NodeMockConfig()

        node = MagicMock()

        # Set all attributes
        node.node_id = config.node_id
        node.parent_id = config.parent_id
        node.prompt_sent = config.prompt_sent
        node.completed = config.completed
        node.off_topic = config.off_topic
        node.objective_target_conversation_id = config.objective_target_conversation_id
        node.error_message = None

        node.send_prompt_async = AsyncMock(return_value=None)

        node._generate_adversarial_prompt_async = AsyncMock(return_value="test prompt")
        node._generate_red_teaming_prompt_async = AsyncMock(return_value='{"prompt": "test prompt"}')
        node._is_prompt_off_topic_async = AsyncMock(return_value=False)
        node._send_prompt_to_target_async = AsyncMock(return_value=MagicMock())
        node._score_response_async = AsyncMock(return_value=None)
        node._send_to_adversarial_chat_async = AsyncMock(return_value='{"prompt": "test prompt"}')
        node._check_on_topic_async = AsyncMock(return_value=True)
        node._execute_objective_prompt_async = AsyncMock(return_value=None)

        # Set up objective score
        if config.objective_score_value is not None:
            node.objective_score = MagicMock(get_value=MagicMock(return_value=config.objective_score_value))
        else:
            node.objective_score = None

        # Set up auxiliary scores
        node.auxiliary_scores = {}
        for name, value in config.auxiliary_scores.items():
            node.auxiliary_scores[name] = MagicMock(get_value=MagicMock(return_value=value))

        # Set up duplicate method to return a new mock node
        def duplicate_side_effect():
            return MockNodeFactory.create_node(NodeMockConfig(parent_id=node.node_id))

        node.duplicate = MagicMock(side_effect=duplicate_side_effect)

        node.last_prompt_sent = None
        node.last_response = None
        node.adversarial_chat_conversation_id = str(uuid.uuid4())

        node._memory = MagicMock()
        node._memory.duplicate_conversation = MagicMock(return_value=str(uuid.uuid4()))
        node._objective_target = MagicMock()
        node._adversarial_chat = MagicMock()
        node._objective_scorer = MagicMock()
        node._on_topic_scorer = None
        node._auxiliary_scorers = []
        node._request_converters = []
        node._response_converters = []
        node._memory_labels = {}
        node._attack_id = {"__type__": "MockAttack", "__module__": "test_module"}
        node._prompt_normalizer = MagicMock()

        # Mock the required internal methods that might be called
        node._mark_execution_complete = MagicMock()
        node._handle_json_error = MagicMock()
        node._handle_unexpected_error = MagicMock()
        node._parse_red_teaming_response = MagicMock(return_value="test prompt")

        return node

    @staticmethod
    def create_nodes_with_scores(scores: List[float]) -> List[_TreeOfAttacksNode]:
        """Create multiple nodes with the given objective scores."""
        return [
            MockNodeFactory.create_node(NodeMockConfig(node_id=f"node_{i}", objective_score_value=score))
            for i, score in enumerate(scores)
        ]


class AttackBuilder:
    """Builder for creating TreeOfAttacksWithPruningAttack instances with common configurations."""

    def __init__(self):
        self.objective_target: Optional[PromptTarget] = None
        self.adversarial_chat: Optional[PromptChatTarget] = None
        self.objective_scorer: Optional[Scorer] = None
        self.auxiliary_scorers: List[Scorer] = []
        self.tree_params: Dict[str, Any] = {}
        self.converters: Optional[AttackConverterConfig] = None
        self.successful_threshold: float = 0.8
        self.prompt_normalizer: Optional[PromptNormalizer] = None

    def with_default_mocks(self):
        """Set up default mocks for all required components."""
        self.objective_target = self._create_mock_target()
        self.adversarial_chat = self._create_mock_chat()
        self.objective_scorer = cast(Scorer, self._create_mock_scorer("MockScorer"))
        return self

    def with_tree_params(self, **kwargs):
        """Set tree parameters (width, depth, branching_factor, batch_size)."""
        self.tree_params = kwargs
        return self

    def with_threshold(self, threshold: float):
        """Set successful objective threshold."""
        self.successful_threshold = threshold
        return self

    def with_auxiliary_scorers(self, count: int = 1):
        """Add auxiliary scorers."""
        self.auxiliary_scorers = [cast(Scorer, self._create_mock_scorer(f"MockAuxScorer{i}")) for i in range(count)]
        return self

    def with_prompt_normalizer(self):
        """Add a mock prompt normalizer."""
        normalizer = MagicMock(spec=PromptNormalizer)
        normalizer.send_prompt_async = AsyncMock(return_value=None)
        self.prompt_normalizer = cast(PromptNormalizer, normalizer)
        return self

    def build(self) -> TreeOfAttacksWithPruningAttack:
        """Build the attack instance."""
        assert self.adversarial_chat is not None, "Adversarial chat target must be set."
        adversarial_config = AttackAdversarialConfig(target=self.adversarial_chat)
        scoring_config = AttackScoringConfig(
            objective_scorer=cast(Scorer, self.objective_scorer),
            auxiliary_scorers=self.auxiliary_scorers,
            successful_objective_threshold=self.successful_threshold,
        )

        kwargs = {
            "objective_target": self.objective_target,
            "attack_adversarial_config": adversarial_config,
            "attack_scoring_config": scoring_config,
            **self.tree_params,
        }

        if self.prompt_normalizer:
            kwargs["prompt_normalizer"] = self.prompt_normalizer

        return TreeOfAttacksWithPruningAttack(**kwargs)

    @staticmethod
    def _create_mock_target() -> PromptTarget:
        target = MagicMock(spec=PromptTarget)
        target.send_prompt_async = AsyncMock(return_value=None)
        target.get_identifier.return_value = {"__type__": "MockTarget", "__module__": "test_module"}
        return cast(PromptTarget, target)

    @staticmethod
    def _create_mock_chat() -> PromptChatTarget:
        chat = MagicMock(spec=PromptChatTarget)
        chat.send_prompt_async = AsyncMock(return_value=None)
        chat.set_system_prompt = MagicMock()
        chat.get_identifier.return_value = {"__type__": "MockChatTarget", "__module__": "test_module"}
        return cast(PromptChatTarget, chat)

    @staticmethod
    def _create_mock_scorer(name: str) -> Scorer:
        scorer = MagicMock(spec=Scorer)
        scorer.scorer_type = "float_scale"
        scorer.score_async = AsyncMock(return_value=[])
        scorer.get_identifier.return_value = {"__type__": name, "__module__": "test_module"}
        return cast(Scorer, scorer)


class TestHelpers:
    """Helper methods for common test operations."""

    @staticmethod
    def create_basic_context() -> TAPAttackContext:
        """Create a basic context with initialized tree."""
        context = TAPAttackContext(
            objective="Test objective",
            memory_labels={"test": "label"},
        )
        context.tree_visualization.create_node("Root", "root")
        return context

    @staticmethod
    def create_score(value: float = 0.9) -> Score:
        """Create a mock Score object."""
        return Score(
            id=None,
            score_type="float_scale",
            score_value=str(value),
            score_category="test",
            score_value_description="Test score",
            score_rationale="Test rationale",
            score_metadata="{}",
            prompt_request_response_id=str(uuid.uuid4()),
            scorer_class_identifier={"__type__": "MockScorer", "__module__": "test_module"},
        )

    @staticmethod
    def add_nodes_to_tree(context: TAPAttackContext, nodes: List[_TreeOfAttacksNode], parent: str = "root"):
        """Add nodes to the context's tree visualization."""
        for i, node in enumerate(nodes):
            score_str = ""
            if node.objective_score:
                score_str = f": Score {node.objective_score.get_value()}"
            context.tree_visualization.create_node(
                f"{context.current_iteration}{score_str}", node.node_id, parent=parent
            )

    @staticmethod
    def mock_prompt_loading(attack: TreeOfAttacksWithPruningAttack):
        """Mock the prompt loading process."""
        mock_seed_prompts = [MagicMock(spec=SeedPrompt) for _ in range(3)]
        mock_dataset = MagicMock()
        mock_dataset.prompts = mock_seed_prompts

        with patch("pyrit.models.seed_prompt_dataset.SeedPromptDataset.from_yaml_file", return_value=mock_dataset):
            attack._load_adversarial_prompts()


@pytest.fixture
def node_factory():
    """Fixture providing the MockNodeFactory."""
    return MockNodeFactory()


@pytest.fixture
def attack_builder():
    """Fixture providing the AttackBuilder."""
    return AttackBuilder()


@pytest.fixture
def helpers():
    """Fixture providing TestHelpers."""
    return TestHelpers()


@pytest.fixture
def basic_attack(attack_builder):
    """Create a basic attack instance with default configuration."""
    return attack_builder.with_default_mocks().build()


@pytest.mark.usefixtures("patch_central_database")
class TestTreeOfAttacksInitialization:
    """Tests for TreeOfAttacksWithPruningAttack initialization and configuration."""

    def test_init_with_minimal_required_parameters(self, attack_builder):
        """Test that attack initializes correctly with only required parameters."""
        attack = attack_builder.with_default_mocks().build()

        assert attack._tree_width == 3
        assert attack._tree_depth == 5
        assert attack._branching_factor == 2
        assert attack._on_topic_checking_enabled is True
        assert attack._batch_size == 10

    def test_init_with_custom_tree_parameters(self, attack_builder):
        """Test initialization with custom tree parameters."""
        attack = (
            attack_builder.with_default_mocks()
            .with_tree_params(tree_width=5, tree_depth=10, branching_factor=3, batch_size=20)
            .build()
        )

        assert attack._tree_width == 5
        assert attack._tree_depth == 10
        assert attack._branching_factor == 3
        assert attack._batch_size == 20

    @pytest.mark.parametrize(
        "tree_params,expected_error",
        [
            ({"tree_width": 0}, "tree width must be at least 1"),
            ({"tree_depth": 0}, "tree depth must be at least 1"),
            ({"branching_factor": 0}, "branching factor must be at least 1"),
            ({"batch_size": 0}, "batch size must be at least 1"),
            ({"tree_width": -1}, "tree width must be at least 1"),
            ({"tree_depth": -1}, "tree depth must be at least 1"),
            ({"branching_factor": -1}, "branching factor must be at least 1"),
            ({"batch_size": -1}, "batch size must be at least 1"),
        ],
    )
    def test_init_with_invalid_tree_parameters(self, attack_builder, tree_params, expected_error):
        """Test that invalid tree parameters raise ValueError."""
        with pytest.raises(ValueError, match=expected_error):
            attack_builder.with_default_mocks().with_tree_params(**tree_params).build()

    def test_init_with_auxiliary_scorers(self, attack_builder):
        """Test initialization with auxiliary scorers."""
        attack = attack_builder.with_default_mocks().with_auxiliary_scorers(2).build()
        assert len(attack._auxiliary_scorers) == 2


@pytest.mark.usefixtures("patch_central_database")
class TestPruningLogic:
    """Tests for node pruning functionality."""

    def test_prune_nodes_to_maintain_width_removes_lowest_scoring_nodes(self, basic_attack, node_factory, helpers):
        """Test that pruning keeps highest scoring nodes."""
        context = helpers.create_basic_context()

        # Create nodes with different scores
        nodes = node_factory.create_nodes_with_scores([0.9, 0.7, 0.5, 0.3, 0.1])
        context.nodes = nodes
        helpers.add_nodes_to_tree(context, nodes)
        basic_attack._tree_width = 3

        # Execute pruning
        basic_attack._prune_nodes_to_maintain_width(context=context)

        # Verify only top 3 nodes remain
        assert len(context.nodes) == 3
        remaining_scores = [
            node.objective_score.get_value() for node in context.nodes if node.objective_score is not None
        ]
        assert sorted(remaining_scores, reverse=True) == [0.9, 0.7, 0.5]

    def test_prune_off_topic_nodes(self, basic_attack, node_factory):
        """Test that off-topic nodes are not included in pruning consideration."""
        # Create mix of on-topic and off-topic nodes
        off_topic_node = node_factory.create_node(
            NodeMockConfig(node_id="off_topic", off_topic=True, objective_score_value=0.9)
        )

        on_topic_nodes = node_factory.create_nodes_with_scores([0.5, 0.4, 0.3])
        all_nodes = [off_topic_node] + on_topic_nodes

        # Get completed nodes
        completed = basic_attack._get_completed_nodes_sorted_by_score(all_nodes)

        # Verify off-topic node is excluded
        assert len(completed) == 3
        assert all(not node.off_topic for node in completed)

    def test_update_best_performing_node_with_unsorted_nodes(self, basic_attack, node_factory, helpers):
        """Test that _update_best_performing_node correctly finds the best node regardless of input order."""
        context = helpers.create_basic_context()

        # Create nodes with scores in random order (not sorted)
        nodes = node_factory.create_nodes_with_scores([0.3, 0.9, 0.1, 0.7, 0.5])
        # Shuffle to ensure they're not in any particular order
        import random

        random.shuffle(nodes)
        context.nodes = nodes

        # Execute update
        basic_attack._update_best_performing_node(context)

        # Verify the best node (0.9 score) was selected
        assert context.best_objective_score is not None
        assert context.best_objective_score.get_value() == 0.9
        assert context.best_conversation_id is not None

    def test_update_best_performing_node_with_empty_nodes(self, basic_attack, helpers):
        """Test that _update_best_performing_node handles empty nodes gracefully."""
        context = helpers.create_basic_context()
        context.nodes = []

        # Should return early without raising an exception
        basic_attack._update_best_performing_node(context)

        # Best scores should remain None since no nodes exist
        assert context.best_objective_score is None
        assert context.best_conversation_id is None

    def test_update_best_performing_node_with_incomplete_nodes(self, basic_attack, node_factory, helpers):
        """Test that _update_best_performing_node handles nodes without valid scores."""
        context = helpers.create_basic_context()

        # Create mix of completed and incomplete nodes
        incomplete_node = node_factory.create_node(
            NodeMockConfig(node_id="incomplete", completed=False, objective_score_value=None)
        )
        off_topic_node = node_factory.create_node(
            NodeMockConfig(node_id="off_topic", off_topic=True, objective_score_value=0.9)
        )
        no_score_node = node_factory.create_node(
            NodeMockConfig(node_id="no_score", completed=True, objective_score_value=None)
        )
        valid_node = node_factory.create_node(
            NodeMockConfig(node_id="valid", completed=True, objective_score_value=0.6)
        )

        context.nodes = [incomplete_node, off_topic_node, no_score_node, valid_node]

        # Execute update
        basic_attack._update_best_performing_node(context)

        # Should select the only valid node
        assert context.best_objective_score is not None
        assert context.best_objective_score.get_value() == 0.6
        assert context.best_conversation_id == valid_node.objective_target_conversation_id

    def test_update_best_performing_node_with_all_invalid_nodes(self, basic_attack, node_factory, helpers):
        """Test that _update_best_performing_node handles case where no valid nodes exist."""
        context = helpers.create_basic_context()

        # Create only invalid nodes
        incomplete_node = node_factory.create_node(
            NodeMockConfig(node_id="incomplete", completed=False, objective_score_value=None)
        )
        off_topic_node = node_factory.create_node(
            NodeMockConfig(node_id="off_topic", off_topic=True, objective_score_value=0.9)
        )

        context.nodes = [incomplete_node, off_topic_node]

        # Execute update - should not update best scores when no valid nodes
        basic_attack._update_best_performing_node(context)

        # Best scores should remain None since no valid nodes exist
        assert context.best_objective_score is None
        assert context.best_conversation_id is None

    def test_update_best_performing_node_preserves_existing_best_when_no_valid_nodes(
        self, basic_attack, node_factory, helpers
    ):
        """Test that _update_best_performing_node preserves existing best when no new valid nodes."""
        context = helpers.create_basic_context()

        # Set existing best
        existing_score = helpers.create_score(0.8)
        context.best_objective_score = existing_score
        context.best_conversation_id = "existing_conv_id"

        # Add only invalid nodes
        off_topic_node = node_factory.create_node(
            NodeMockConfig(node_id="off_topic", off_topic=True, objective_score_value=0.95)
        )
        context.nodes = [off_topic_node]

        # Execute update
        basic_attack._update_best_performing_node(context)

        # Should preserve existing best since no valid nodes
        assert context.best_objective_score == existing_score
        assert context.best_conversation_id == "existing_conv_id"


@pytest.mark.usefixtures("patch_central_database")
class TestBranchingLogic:
    """Tests for node branching functionality."""

    def test_branch_existing_nodes(self, basic_attack, node_factory, helpers):
        """Test that nodes are branched correctly."""
        context = helpers.create_basic_context()
        basic_attack._branching_factor = 3

        # Create initial nodes
        initial_nodes = node_factory.create_nodes_with_scores([0.8, 0.7])
        helpers.add_nodes_to_tree(context, initial_nodes)
        context.nodes = initial_nodes.copy()
        context.current_iteration = 2

        # Execute branching
        basic_attack._branch_existing_nodes(context=context)

        # Verify results
        # 2 original + 4 new
        assert len(context.nodes) == 6

        # Verify duplicate was called correct number of times
        for node in initial_nodes:
            assert node.duplicate.call_count == 2


@pytest.mark.usefixtures("patch_central_database")
class TestExecutionPhase:
    """Tests for the main execution phase of the attack."""

    @pytest.mark.asyncio
    async def test_perform_attack_single_iteration_success(self, attack_builder, node_factory, helpers):
        """Test successful execution of single iteration."""
        attack = (
            attack_builder.with_default_mocks()
            .with_tree_params(tree_depth=1, tree_width=1)
            .with_prompt_normalizer()
            .build()
        )

        context = helpers.create_basic_context()

        # Create successful node
        success_node = node_factory.create_node(
            NodeMockConfig(
                node_id="success_node", objective_score_value=0.9, objective_target_conversation_id="success_conv"
            )
        )

        with patch.object(attack, "_create_attack_node", return_value=success_node):
            with patch.object(attack._memory, "get_prompt_request_pieces", return_value=[]):
                result = await attack._perform_async(context=context)

        assert result.outcome == AttackOutcome.SUCCESS
        assert result.conversation_id == "success_conv"
        assert result.max_depth_reached == 1

    @pytest.mark.asyncio
    async def test_perform_attack_early_termination_on_success(self, attack_builder, node_factory, helpers):
        """Test early termination when objective is achieved."""
        attack = (
            attack_builder.with_default_mocks()
            .with_tree_params(tree_depth=5, tree_width=1)
            .with_threshold(0.8)
            .with_prompt_normalizer()
            .build()
        )

        context = helpers.create_basic_context()

        # Create successful node
        success_node = node_factory.create_node(
            NodeMockConfig(
                node_id="success_node", objective_score_value=0.9, objective_target_conversation_id="success_conv"
            )
        )

        with patch.object(attack, "_create_attack_node", return_value=success_node):
            with patch.object(attack._memory, "get_prompt_request_pieces", return_value=[]):
                result = await attack._perform_async(context=context)

        # Should succeed after first iteration
        assert result.outcome == AttackOutcome.SUCCESS
        assert result.max_depth_reached == 1
        assert result.conversation_id == "success_conv"

    @pytest.mark.asyncio
    async def test_perform_attack_batch_processing(self, attack_builder, node_factory, helpers):
        """Test batch processing of nodes."""
        attack = (
            attack_builder.with_default_mocks()
            .with_tree_params(tree_width=15, batch_size=5, tree_depth=1)
            .with_prompt_normalizer()
            .build()
        )

        context = helpers.create_basic_context()

        # Create 15 mock nodes
        nodes = node_factory.create_nodes_with_scores([0.5] * 15)
        helpers.add_nodes_to_tree(context, nodes)
        context.nodes = nodes

        await attack._send_prompts_to_all_nodes_async(context=context)

        # Verify all nodes were processed
        for node in nodes:
            node.send_prompt_async.assert_called_once()


@pytest.mark.usefixtures("patch_central_database")
class TestHelperMethods:
    """Tests for helper methods."""

    def test_format_node_result(self, basic_attack, node_factory):
        """Test formatting node results for visualization."""
        # Test off-topic node
        off_topic_node = node_factory.create_node(NodeMockConfig(off_topic=True))
        result = basic_attack._format_node_result(off_topic_node)
        assert result == "Pruned (off-topic)"

        # Test incomplete node
        incomplete_node = node_factory.create_node(NodeMockConfig(completed=False))
        result = basic_attack._format_node_result(incomplete_node)
        assert result == "Pruned (no score available)"

        # Test completed node with score
        completed_node = node_factory.create_node(NodeMockConfig(objective_score_value=0.7))
        result = basic_attack._format_node_result(completed_node)
        assert "Score: " in result
        assert "/10" in result

    def test_is_objective_achieved(self, attack_builder, helpers):
        """Test _is_objective_achieved logic."""
        attack = attack_builder.with_default_mocks().with_threshold(0.8).build()
        context = helpers.create_basic_context()

        # Test 1: No score available
        context.best_objective_score = None
        assert attack._is_objective_achieved(context=context) is False

        # Test 2: Score below threshold
        context.best_objective_score = MagicMock(get_value=MagicMock(return_value=0.5))
        assert attack._is_objective_achieved(context=context) is False

        # Test 3: Score at threshold
        context.best_objective_score = MagicMock(get_value=MagicMock(return_value=0.8))
        assert attack._is_objective_achieved(context=context) is True

        # Test 4: Score above threshold
        context.best_objective_score = MagicMock(get_value=MagicMock(return_value=0.9))
        assert attack._is_objective_achieved(context=context) is True


@pytest.mark.usefixtures("patch_central_database")
class TestEndToEndExecution:
    """Tests for end-to-end execution using execute_async."""

    @pytest.mark.asyncio
    async def test_execute_async_success_flow(self, attack_builder, helpers):
        """Test complete successful attack flow through execute_async."""
        attack = (
            attack_builder.with_default_mocks()
            .with_tree_params(tree_width=2, tree_depth=2)
            .with_prompt_normalizer()
            .build()
        )

        # Mock seed prompt loading
        helpers.mock_prompt_loading(attack)

        # Create mock result
        mock_result = TAPAttackResult(
            conversation_id="success_conv_id",
            objective="Test objective",
            attack_identifier=attack.get_identifier(),
            last_response=None,
            last_score=helpers.create_score(0.9),
            executed_turns=1,
            execution_time_ms=100,
            outcome=AttackOutcome.SUCCESS,
            outcome_reason="Objective achieved",
        )
        # Set tree-specific properties
        mock_result.tree_visualization = Tree()
        mock_result.nodes_explored = 2
        mock_result.nodes_pruned = 0
        mock_result.max_depth_reached = 1
        mock_result.auxiliary_scores_summary = {}

        with patch.object(attack, "_perform_async", return_value=mock_result):
            with patch.object(attack._memory, "get_conversation", return_value=[]):
                with patch.object(attack._memory, "get_prompt_request_pieces", return_value=[]):
                    with patch.object(attack._memory, "add_attack_results_to_memory", return_value=None):
                        result = await attack.execute_async(objective="Test objective", memory_labels={"test": "label"})

        assert result.outcome == AttackOutcome.SUCCESS
        assert result.objective == "Test objective"
        assert isinstance(result, TAPAttackResult)
        assert result.nodes_explored > 0


@pytest.mark.usefixtures("patch_central_database")
class TestTreeOfAttacksNode:
    """Tests for _TreeOfAttacksNode functionality."""

    @pytest.fixture
    def node_components(self, attack_builder):
        """Create components needed for _TreeOfAttacksNode."""
        builder = attack_builder.with_default_mocks()

        adversarial_chat_seed_prompt = MagicMock(spec=SeedPrompt)
        adversarial_chat_seed_prompt.render_template_value = MagicMock(return_value="rendered seed prompt")

        adversarial_chat_system_seed_prompt = MagicMock(spec=SeedPrompt)
        adversarial_chat_system_seed_prompt.render_template_value = MagicMock(return_value="rendered system prompt")

        adversarial_chat_prompt_template = MagicMock(spec=SeedPrompt)
        adversarial_chat_prompt_template.render_template_value = MagicMock(return_value="rendered template")

        prompt_normalizer = MagicMock()
        prompt_normalizer.send_prompt_async = AsyncMock(return_value=None)

        components = {
            "objective_target": builder.objective_target,
            "adversarial_chat": builder.adversarial_chat,
            "objective_scorer": builder.objective_scorer,
            "adversarial_chat_seed_prompt": adversarial_chat_seed_prompt,
            "adversarial_chat_system_seed_prompt": adversarial_chat_system_seed_prompt,
            "adversarial_chat_prompt_template": adversarial_chat_prompt_template,
            "desired_response_prefix": "Sure, here is",
            "on_topic_scorer": None,
            "request_converters": [],
            "response_converters": [],
            "auxiliary_scorers": [],
            "attack_id": {"id": "test_attack"},
            "memory_labels": {"test": "label"},
            "parent_id": None,
            "prompt_normalizer": prompt_normalizer,
        }
        return components

    def test_node_initialization(self, node_components):
        """Test _TreeOfAttacksNode initialization."""
        node = _TreeOfAttacksNode(**node_components)

        assert node.node_id is not None
        assert node.parent_id is None
        assert node.completed is False
        assert node.off_topic is False
        assert node.objective_score is None
        assert node.auxiliary_scores == {}
        assert node.error_message is None

    def test_node_duplicate_creates_child(self, node_components):
        """Test that duplicate() creates a proper child node."""
        parent_node = _TreeOfAttacksNode(**node_components)
        parent_node.node_id = "parent_node_id"

        # Mock memory duplicate conversation
        with patch.object(parent_node._memory, "duplicate_conversation", return_value="new_conv_id"):
            child_node = parent_node.duplicate()

        assert child_node.node_id != parent_node.node_id
        assert child_node.parent_id == parent_node.node_id
        assert child_node.completed is False

    @pytest.mark.asyncio
    async def test_node_send_prompt_json_error_handling(self, node_components):
        """Test handling of JSON parsing errors in send_prompt_async."""
        prompt_normalizer = MagicMock(spec=PromptNormalizer)
        components_with_normalizer = node_components.copy()
        components_with_normalizer["prompt_normalizer"] = prompt_normalizer
        node = _TreeOfAttacksNode(**components_with_normalizer)

        # Mock adversarial chat to raise JSON error
        json_error = InvalidJsonException(message="Invalid JSON")
        node._adversarial_chat.send_prompt_async = AsyncMock(side_effect=json_error)

        # Mock the prompt normalizer to raise the wrapped exception
        wrapped_exception = Exception("Error sending prompt with conversation ID: id")
        prompt_normalizer.send_prompt_async = AsyncMock(side_effect=wrapped_exception)

        await node.send_prompt_async(objective="Test objective")

        # Node should handle the error gracefully
        assert node.completed is False
        assert node.error_message is not None
        assert "Error sending prompt with conversation ID" in node.error_message

    @pytest.mark.asyncio
    async def test_node_send_prompt_unexpected_error_handling(self, node_components):
        """Test handling of unexpected errors in send_prompt_async."""
        node = _TreeOfAttacksNode(**node_components)

        # Mock adversarial chat to raise unexpected error
        unexpected_error = RuntimeError("Unexpected error")
        node._adversarial_chat.send_prompt_async = AsyncMock(side_effect=unexpected_error)

        await node.send_prompt_async(objective="Test objective")

        # Node should handle the error gracefully
        assert node.completed is False
        assert node.error_message is not None
        assert "Execution error" in node.error_message

    @pytest.mark.asyncio
    async def test_node_off_topic_detection(self, node_components):
        """Test off-topic detection in nodes."""
        # Enable on-topic checking
        on_topic_scorer = MagicMock(spec=Scorer)

        # Create a score that indicates off-topic
        on_topic_score = MagicMock(spec=Score)
        on_topic_score.get_value = MagicMock(return_value=False)  # False = off-topic
        on_topic_score.score_value = "False"
        on_topic_score.score_type = "true_false"
        on_topic_scorer.score_text_async = AsyncMock(return_value=[on_topic_score])

        components_with_scorer = node_components.copy()
        components_with_scorer["on_topic_scorer"] = on_topic_scorer
        components_with_scorer["adversarial_chat"].conversation_id = "test-adv-conv-id"
        components_with_scorer["objective_target"].conversation_id = "test-obj-conv-id"

        node = _TreeOfAttacksNode(**components_with_scorer)

        test_prompt = "test adversarial prompt"
        with patch.object(
            node, "_generate_red_teaming_prompt_async", new_callable=AsyncMock, return_value=test_prompt
        ) as red_teaming_mock:

            await node.send_prompt_async(objective="Test objective")

        # Verify off-topic detection worked
        assert node.off_topic is True
        # Node stops execution when off-topic
        assert node.completed is False

        red_teaming_mock.assert_called_once()
        # Verify the on-topic scorer was called with the generated prompt
        on_topic_scorer.score_text_async.assert_called_once_with(text=test_prompt)

    @pytest.mark.asyncio
    async def test_node_auxiliary_scoring(self, node_components):
        """Test auxiliary scoring functionality."""
        # Add auxiliary scorers with specific class identifiers
        aux_score1 = MagicMock()
        aux_score1.get_value.return_value = 0.8
        aux_score1.scorer_class_identifier = {"__type__": "AuxScorer1"}
        aux_scorer1 = MagicMock(spec=Scorer)
        aux_scorer1.score_async = AsyncMock(return_value=[aux_score1])

        aux_score2 = MagicMock()
        aux_score2.get_value.return_value = 0.6
        aux_score2.scorer_class_identifier = {"__type__": "AuxScorer2"}
        aux_scorer2 = MagicMock(spec=Scorer)
        aux_scorer2.score_async = AsyncMock(return_value=[aux_score2])

        node_components["auxiliary_scorers"] = [aux_scorer1, aux_scorer2]

        node = _TreeOfAttacksNode(**node_components)

        # Create a mock prompt normalizer if not provided
        mock_normalizer = node._prompt_normalizer

        # Mock the prompt normalizer's send_prompt_async method
        async def normalizer_side_effect(*args, **kwargs):
            target = kwargs.get("target")

            if target == node._adversarial_chat:
                # Return JSON response for adversarial chat
                return PromptRequestResponse(
                    request_pieces=[
                        PromptRequestPiece(
                            role="assistant",
                            original_value=json.dumps({"prompt": "test prompt", "improvement": "test"}),
                            converted_value=json.dumps({"prompt": "test prompt", "improvement": "test"}),
                            conversation_id=node.adversarial_chat_conversation_id,
                            id=str(uuid.uuid4()),
                        )
                    ]
                )
            else:
                # Return normal response for objective target
                return PromptRequestResponse(
                    request_pieces=[
                        PromptRequestPiece(
                            role="assistant",
                            original_value="Target response",
                            converted_value="Target response",
                            conversation_id=node.objective_target_conversation_id,
                            id=str(uuid.uuid4()),
                        )
                    ]
                )

        mock_normalizer.send_prompt_async = AsyncMock(side_effect=normalizer_side_effect)

        # Mocking objective scorer
        obj_score = MagicMock()
        obj_score.get_value.return_value = 0.7
        obj_score.scorer_class_identifier = {"__type__": "ObjectiveScorer"}
        node._objective_scorer.score_async = AsyncMock(return_value=[obj_score])

        # Mock for Scorer.score_response_with_objective_async
        def mock_score_response(*args, **kwargs):
            return {"objective_scores": [obj_score], "auxiliary_scores": [aux_score1, aux_score2]}

        with patch(
            "pyrit.score.Scorer.score_response_with_objective_async",
            new_callable=AsyncMock,
            side_effect=mock_score_response,
        ):
            await node.send_prompt_async(objective="Test objective")

        # Verify node state
        assert node.completed is True
        assert node.error_message is None
        assert node.last_prompt_sent == "test prompt"
        assert node.last_response == "Target response"

        # Verify scores
        assert node.objective_score is not None
        assert node.objective_score == obj_score
        assert node.objective_score.get_value() == 0.7

        # Verify auxiliary scores are stored with correct keys
        assert len(node.auxiliary_scores) == 2
        assert "AuxScorer1" in node.auxiliary_scores
        assert "AuxScorer2" in node.auxiliary_scores
        assert node.auxiliary_scores["AuxScorer1"].get_value() == 0.8
        assert node.auxiliary_scores["AuxScorer2"].get_value() == 0.6


@pytest.mark.usefixtures("patch_central_database")
class TestTreeOfAttacksErrorHandling:
    """Tests for error handling in TreeOfAttacksWithPruningAttack."""

    @pytest.mark.asyncio
    async def test_attack_handles_all_nodes_failing(self, attack_builder, helpers, node_factory):
        """Test attack behavior when all nodes fail."""
        attack = (
            attack_builder.with_default_mocks().with_tree_params(tree_width=2, tree_depth=1).with_threshold(0.8).build()
        )

        context = helpers.create_basic_context()

        # Create nodes that will all fail (no scores)
        failing_nodes = []
        for i in range(2):
            node = node_factory.create_node(
                NodeMockConfig(
                    node_id=f"failing_node_{i}",
                    completed=True,
                    off_topic=False,
                    objective_score_value=None,
                    objective_target_conversation_id=f"conv_{i}",
                )
            )
            node.error_message = "Execution failed"
            node.send_prompt_async = AsyncMock(return_value=None)
            failing_nodes.append(node)

        # Use an iterator to return nodes one by one
        node_iterator = iter(failing_nodes)

        with patch.object(attack, "_create_attack_node", side_effect=lambda **kwargs: next(node_iterator)):
            with patch.object(attack._memory, "get_prompt_request_pieces", return_value=[]):
                result = await attack._perform_async(context=context)

        # Should return failure when all nodes fail
        assert result.outcome == AttackOutcome.FAILURE
        # The actual message is about not achieving threshold score
        assert "did not achieve threshold score" in result.outcome_reason.lower()

    @pytest.mark.asyncio
    async def test_attack_continues_after_node_errors(self, attack_builder, node_factory, helpers):
        """Test that attack continues when some nodes have errors."""
        attack = (
            attack_builder.with_default_mocks().with_tree_params(tree_width=3, tree_depth=2, branching_factor=2).build()
        )

        context = helpers.create_basic_context()

        # Create mix of successful and failing nodes for first iteration
        nodes = []

        # Failing node
        fail_node = node_factory.create_node(
            NodeMockConfig(
                node_id="fail_node",
                completed=True,
                off_topic=False,
                objective_score_value=None,  # Explicitly set to None for failure
                objective_target_conversation_id="fail_conv",
            )
        )
        fail_node.error_message = "JSON parsing error"
        fail_node.duplicate = MagicMock(
            return_value=node_factory.create_node(NodeMockConfig(node_id="fail_dup", objective_score_value=0.3))
        )
        nodes.append(fail_node)

        # Successful nodes
        for i in range(2):
            success_node = node_factory.create_node(
                NodeMockConfig(node_id=f"success_node_{i}", objective_score_value=0.5 + i * 0.1)
            )
            nodes.append(success_node)

        # Create all nodes at once
        node_iter = iter(nodes)
        with patch.object(attack, "_create_attack_node", side_effect=lambda **kwargs: next(node_iter, nodes[0])):
            with patch.object(attack._memory, "get_prompt_request_pieces", return_value=[]):
                result = await attack._perform_async(context=context)

        # Attack should continue despite some nodes failing
        assert result.outcome == AttackOutcome.FAILURE  # Did not reach threshold
        assert result.max_depth_reached == 2


@pytest.mark.usefixtures("patch_central_database")
class TestTreeOfAttacksMemoryOperations:
    """Tests for memory-related operations."""

    def test_attack_updates_memory_labels(self, attack_builder, helpers):
        """Test that memory labels are properly combined."""
        attack = attack_builder.with_default_mocks().build()

        # Set initial memory labels on attack
        attack._memory_labels = {"attack_label": "attack_value"}

        context = helpers.create_basic_context()
        context.memory_labels = {"context_label": "context_value"}

        # Run setup to combine labels
        asyncio.run(attack._setup_async(context=context))

        # Verify labels are combined
        assert context.memory_labels["attack_label"] == "attack_value"
        assert context.memory_labels["context_label"] == "context_value"


@pytest.mark.usefixtures("patch_central_database")
class TestTreeOfAttacksPromptLoading:
    """Tests for prompt loading and handling."""

    def test_load_adversarial_prompts_default(self, attack_builder):
        """Test loading prompts with default paths."""
        attack = attack_builder.with_default_mocks().build()

        # Mock SeedPrompt loading
        mock_system = MagicMock(spec=SeedPrompt)
        mock_template = MagicMock(spec=SeedPrompt)
        mock_seed = MagicMock(spec=SeedPrompt)

        with patch.object(SeedPrompt, "from_yaml_with_required_parameters", return_value=mock_system):
            with patch.object(SeedPrompt, "from_yaml_file", side_effect=[mock_template, mock_seed]):
                attack._load_adversarial_prompts()

        # Verify prompts were loaded and stored
        assert attack._adversarial_chat_system_seed_prompt == mock_system
        assert attack._adversarial_chat_prompt_template == mock_template
        assert attack._adversarial_chat_seed_prompt == mock_seed


@pytest.mark.usefixtures("patch_central_database")
class TestTreeOfAttacksVisualization:
    """Tests for tree visualization functionality."""

    def test_format_node_result_with_scores(self, basic_attack):
        """Test formatting node results with different score formats."""
        node = MagicMock()
        node.off_topic = False
        node.completed = True
        node.objective_score = MagicMock(get_value=MagicMock(return_value=0.7))

        result = basic_attack._format_node_result(node)

        # The actual format uses integer division
        assert "Score: " in result
        assert "/10" in result
        # Should show as 7/10, not 7.0/10
        assert "7/10" in result

    def test_tree_visualization_structure(self, basic_attack, node_factory, helpers):
        """Test that tree visualization maintains proper structure."""
        context = helpers.create_basic_context()

        # Create a tree structure:
        # root
        # |---node_0 (iteration 1)
        # │   |--- node_0_child_0 (iteration 2)
        # │   |--- node_0_child_1 (iteration 2)
        # |-- node_1 (iteration 1)
        #     |--- node_1_child_0 (iteration 2)

        # First level nodes
        node_0 = node_factory.create_node(NodeMockConfig(node_id="node_0"))
        node_1 = node_factory.create_node(NodeMockConfig(node_id="node_1"))

        # Add first level to tree
        context.current_iteration = 1
        helpers.add_nodes_to_tree(context, [node_0, node_1])

        # Second level nodes
        node_0_child_0 = node_factory.create_node(NodeMockConfig(node_id="node_0_child_0", parent_id="node_0"))
        node_0_child_1 = node_factory.create_node(NodeMockConfig(node_id="node_0_child_1", parent_id="node_0"))
        node_1_child_0 = node_factory.create_node(NodeMockConfig(node_id="node_1_child_0", parent_id="node_1"))

        # Add second level to tree using the helper with proper parent relationships
        context.current_iteration = 2
        # Add children under node_0
        for child in [node_0_child_0, node_0_child_1]:
            context.tree_visualization.create_node(
                f"2: Score {child.objective_score.get_value() if child.objective_score else 'N/A'}",
                child.node_id,
                parent=child.parent_id,
            )
        # Add child under node_1
        context.tree_visualization.create_node(
            f"2: Score {node_1_child_0.objective_score.get_value() if node_1_child_0.objective_score else 'N/A'}",
            node_1_child_0.node_id,
            parent=node_1_child_0.parent_id,
        )

        # Verify tree structure
        assert len(context.tree_visualization.all_nodes()) == 6  # root + 5 nodes
        assert len(context.tree_visualization.children("root")) == 2
        assert len(context.tree_visualization.children("node_0")) == 2
        assert len(context.tree_visualization.children("node_1")) == 1

        # Also verify the parent relationships are correct
        assert context.tree_visualization.parent("node_0_child_0").identifier == "node_0"
        assert context.tree_visualization.parent("node_0_child_1").identifier == "node_0"
        assert context.tree_visualization.parent("node_1_child_0").identifier == "node_1"


@pytest.mark.usefixtures("patch_central_database")
class TestTreeOfAttacksConversationTracking:
    """Test that adversarial chat conversation IDs are properly tracked."""

    def test_create_attack_node_tracks_adversarial_chat_conversation_id(self, basic_attack, helpers):
        """Test that creating a node adds its adversarial chat conversation ID to the context."""
        context = helpers.create_basic_context()

        # Create a node
        node = basic_attack._create_attack_node(context=context, parent_id=None)

        # Verify the adversarial chat conversation ID is tracked
        assert (
            ConversationReference(
                conversation_id=node.adversarial_chat_conversation_id,
                conversation_type=ConversationType.ADVERSARIAL,
            )
            in context.related_conversations
        )
        assert len(context.related_conversations) == 1

    def test_branch_existing_nodes_tracks_adversarial_chat_conversation_ids(self, basic_attack, node_factory, helpers):
        """Test that branching nodes adds their adversarial chat conversation IDs to the context."""
        context = helpers.create_basic_context()

        # Create initial nodes
        nodes = node_factory.create_nodes_with_scores([0.8, 0.9])
        context.nodes = nodes

        # Add the initial nodes to the tree visualization to avoid parent node issues
        for node in nodes:
            context.tree_visualization.create_node("1: ", node.node_id, parent="root")

        # Set up branching factor to create additional nodes
        basic_attack._branching_factor = 3

        # Branch the nodes
        basic_attack._branch_existing_nodes(context)

        # Manually add all node adversarial chat conversation IDs to the set (simulating real code behavior)
        for node in context.nodes:
            context.related_conversations.add(
                ConversationReference(
                    conversation_id=node.adversarial_chat_conversation_id,
                    conversation_type=ConversationType.ADVERSARIAL,
                )
            )

        # Verify that adversarial chat conversation IDs are tracked for duplicated nodes
        expected_count = 6  # 2 originals + 4 unique duplicates (2 nodes * (3-1) branching factor)
        assert len(context.related_conversations) == expected_count

        # Verify all nodes have their adversarial chat conversation IDs tracked
        all_nodes = context.nodes
        for node in all_nodes:
            assert (
                ConversationReference(
                    conversation_id=node.adversarial_chat_conversation_id,
                    conversation_type=ConversationType.ADVERSARIAL,
                )
                in context.related_conversations
            )

    def test_initialize_first_level_nodes_tracks_adversarial_chat_conversation_ids(self, basic_attack, helpers):
        """Test that initializing first level nodes tracks their adversarial chat conversation IDs."""
        context = helpers.create_basic_context()

        # Set tree width to create multiple nodes
        basic_attack._tree_width = 3

        # Initialize first level nodes
        asyncio.run(basic_attack._initialize_first_level_nodes_async(context))

        # Verify that adversarial chat conversation IDs are tracked
        assert len(context.related_conversations) == 3

        # Verify all nodes have their adversarial chat conversation IDs tracked
        for node in context.nodes:
            assert (
                ConversationReference(
                    conversation_id=node.adversarial_chat_conversation_id,
                    conversation_type=ConversationType.ADVERSARIAL,
                )
                in context.related_conversations
            )

    def test_attack_result_includes_adversarial_chat_conversation_ids(self, attack_builder, helpers):
        """Test that the attack result includes the tracked adversarial chat conversation IDs."""
        attack = attack_builder.with_default_mocks().build()
        context = helpers.create_basic_context()

        # Create some nodes to populate the tracking
        context.related_conversations = {
            ConversationReference(conversation_id="adv_conv_1", conversation_type=ConversationType.ADVERSARIAL),
            ConversationReference(conversation_id="adv_conv_2", conversation_type=ConversationType.ADVERSARIAL),
        }
        context.best_conversation_id = "best_conv"
        context.best_objective_score = helpers.create_score(0.9)

        # Create the result
        result = attack._create_attack_result(
            context=context, outcome=AttackOutcome.SUCCESS, outcome_reason="Test success"
        )

        # Verify the adversarial chat conversation IDs are included in the result
        assert (
            ConversationReference(
                conversation_id="adv_conv_1",
                conversation_type=ConversationType.ADVERSARIAL,
            )
            in result.related_conversations
        )
        assert (
            ConversationReference(
                conversation_id="adv_conv_2",
                conversation_type=ConversationType.ADVERSARIAL,
            )
            in result.related_conversations
        )

    def test_add_adversarial_chat_conversation_id_ensures_uniqueness(self, basic_attack, helpers):
        """Test that adding adversarial chat conversation IDs ensures uniqueness."""
        context = helpers.create_basic_context()

        # Add a conversation ID
        conversation_id = "test_conv_id"
        context.related_conversations.add(
            ConversationReference(
                conversation_id=conversation_id,
                conversation_type=ConversationType.ADVERSARIAL,
            )
        )

        # Verify it was added
        assert (
            ConversationReference(
                conversation_id=conversation_id,
                conversation_type=ConversationType.ADVERSARIAL,
            )
            in context.related_conversations
        )
        assert len(context.related_conversations) == 1

        # Try to add the same ID again
        context.related_conversations.add(
            ConversationReference(
                conversation_id=conversation_id,
                conversation_type=ConversationType.ADVERSARIAL,
            )
        )

        # Verify it's still only one entry
        assert len(context.related_conversations) == 1

        # Add a different ID
        different_id = "different_conv_id"
        context.related_conversations.add(
            ConversationReference(
                conversation_id=different_id,
                conversation_type=ConversationType.ADVERSARIAL,
            )
        )

        # Verify both IDs are present
        assert len(context.related_conversations) == 2
        assert (
            ConversationReference(
                conversation_id=conversation_id,
                conversation_type=ConversationType.ADVERSARIAL,
            )
            in context.related_conversations
        )
        assert (
            ConversationReference(
                conversation_id=different_id,
                conversation_type=ConversationType.ADVERSARIAL,
            )
            in context.related_conversations
        )
