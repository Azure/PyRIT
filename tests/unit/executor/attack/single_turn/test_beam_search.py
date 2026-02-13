# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import uuid
from unittest.mock import AsyncMock, MagicMock

import pytest
from unit.mocks import get_mock_scorer_identifier, get_mock_target_identifier

from pyrit.executor.attack import (
    AttackScoringConfig,
)
from pyrit.executor.attack.single_turn.beam_search import Beam, BeamSearchAttack, TopKBeamReviewer
from pyrit.prompt_target import OpenAIResponseTarget, PromptTarget
from pyrit.score import FloatScaleScorer, Scorer, TrueFalseScorer


@pytest.fixture
def mock_target():
    """Create a mock prompt target for testing"""
    target = MagicMock(spec=OpenAIResponseTarget)
    target.send_prompt_async = AsyncMock()
    target.get_identifier.return_value = get_mock_target_identifier("MockTarget")
    return target


@pytest.fixture
def mock_bad_target():
    """Create a mock prompt target for testing"""
    target = MagicMock(spec=PromptTarget)
    target.send_prompt_async = AsyncMock()
    target.get_identifier.return_value = get_mock_target_identifier("MockBadTarget")
    return target


@pytest.fixture
def mock_true_false_scorer():
    """Create a mock true/false scorer for testing"""
    scorer = MagicMock(spec=TrueFalseScorer)
    scorer.score_text_async = AsyncMock()
    scorer.get_identifier.return_value = get_mock_scorer_identifier()
    return scorer


@pytest.fixture
def mock_float_scale_scorer():
    """Create a mock float scale scorer for testing"""
    scorer = MagicMock(spec=FloatScaleScorer)
    scorer.score_text_async = AsyncMock()
    scorer.get_identifier.return_value = get_mock_scorer_identifier()
    return scorer


@pytest.fixture
def mock_non_true_false_scorer():
    """Create a mock scorer that is not a true/false type"""
    scorer = MagicMock(spec=Scorer)
    scorer.get_identifier.return_value = get_mock_scorer_identifier()
    return scorer


class TestBeam:
    @pytest.mark.parametrize("n_extend", [1, 2, 4])
    def test_grammar_smoke(self, n_extend):
        beam = Beam(id=str(uuid.uuid4()), text="beam1", score=0.9)

        expected_grammar = f"""
start: PREFIX CONTINUATION
PREFIX: "beam1"
CONTINUATION: /.{{0,{n_extend}}}/
"""
        assert beam.get_grammar(n_chars=n_extend) == expected_grammar

    def test_grammar_with_newline(self):
        beam = Beam(id=str(uuid.uuid4()), text="beam1\nbeam2", score=0.9)

        expected_grammar = """
start: PREFIX CONTINUATION
PREFIX: "beam1\\nbeam2"
CONTINUATION: /.{0,1}/
"""
        assert beam.get_grammar(n_chars=1) == expected_grammar


class TestTopKBeamReviewer:
    @pytest.mark.parametrize("k", [0, -1])
    def test_init_throws_for_non_positive_k(self, k: int):
        with pytest.raises(ValueError, match="k must be a positive integer"):
            _ = TopKBeamReviewer(k=k, drop_chars=0)

    @pytest.mark.parametrize("drop_chars", [-1, -2])
    def test_init_throws_for_negative_drop_chars(self, drop_chars: int):
        with pytest.raises(ValueError, match="drop_chars must be a non-negative integer"):
            _ = TopKBeamReviewer(k=1, drop_chars=drop_chars)

    def test_review_k2d0(self):
        beam1 = Beam(id=str(uuid.uuid4()), text="beam1", score=0.9)
        beam2 = Beam(id=str(uuid.uuid4()), text="beam2", score=0.8)
        beam3 = Beam(id=str(uuid.uuid4()), text="beam3", score=0.7)
        beams = [beam1, beam2, beam3]
        reviewer = TopKBeamReviewer(k=2, drop_chars=0)
        top_k_beams = reviewer.review(beams)
        assert len(top_k_beams) == 3
        assert top_k_beams[0].text == "beam1"
        assert top_k_beams[1].text == "beam2"
        assert top_k_beams[2].text == "beam1"

    def test_review_k2d1(self):
        beam1 = Beam(id=str(uuid.uuid4()), text="beam11", score=0.9)
        beam2 = Beam(id=str(uuid.uuid4()), text="beam22", score=0.8)
        beam3 = Beam(id=str(uuid.uuid4()), text="beam33", score=0.7)
        beam4 = Beam(id=str(uuid.uuid4()), text="beam44", score=0.6)
        beam5 = Beam(id=str(uuid.uuid4()), text="beam55", score=0.5)
        beams = [beam1, beam2, beam3, beam4, beam5]
        reviewer = TopKBeamReviewer(k=2, drop_chars=1)
        top_k_beams = reviewer.review(beams)
        assert len(top_k_beams) == 5
        assert top_k_beams[0].text == "beam11"
        assert top_k_beams[1].text == "beam22"
        assert top_k_beams[2].text == "beam1"
        assert top_k_beams[3].text == "beam2"
        assert top_k_beams[4].text == "beam1"

    def test_review_k1d2(self):
        beam1 = Beam(id=str(uuid.uuid4()), text="beam111", score=0.9)
        beam2 = Beam(id=str(uuid.uuid4()), text="beam222", score=0.8)
        beam3 = Beam(id=str(uuid.uuid4()), text="beam333", score=0.7)
        beams = [beam1, beam2, beam3]
        reviewer = TopKBeamReviewer(k=1, drop_chars=2)
        top_k_beams = reviewer.review(beams)
        assert len(top_k_beams) == 3
        assert top_k_beams[0].text == "beam111"
        assert top_k_beams[1].text == "beam1"
        assert top_k_beams[2].text == "beam1"


@pytest.mark.usefixtures("patch_central_database")
class TestBeamSearchAttack:
    def test_init_with_invalid_target(self, mock_bad_target):
        """Test initialization with an invalid target type"""
        with pytest.raises(
            ValueError, match="BeamSearchAttack requires an OpenAIResponseTarget as the objective target"
        ):
            _ = BeamSearchAttack(objective_target=mock_bad_target, beam_reviewer=TopKBeamReviewer(k=2, drop_chars=1))

    def test_init_without_auxiliary_scorers(self, mock_target, mock_true_false_scorer):
        """Test initialization without auxiliary scorers"""
        scoring_config = AttackScoringConfig(objective_scorer=mock_true_false_scorer)
        with pytest.raises(ValueError, match="BeamSearchAttack requires at least one auxiliary scorer"):
            _ = BeamSearchAttack(
                objective_target=mock_target,
                beam_reviewer=TopKBeamReviewer(k=2, drop_chars=1),
                attack_scoring_config=scoring_config,
            )

    def test_init_invalid_auxiliary_scorers(self, mock_target, mock_true_false_scorer):
        """Test initialization with invalid auxiliary scorers"""
        scoring_config = AttackScoringConfig(
            objective_scorer=mock_true_false_scorer, auxiliary_scorers=[mock_true_false_scorer]
        )
        with pytest.raises(
            ValueError, match="BeamSearchAttack requires all auxiliary scorers to be instances of FloatScaleScorer"
        ):
            _ = BeamSearchAttack(
                objective_target=mock_target,
                beam_reviewer=TopKBeamReviewer(k=2, drop_chars=1),
                attack_scoring_config=scoring_config,
            )
