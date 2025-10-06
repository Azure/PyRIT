# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from pyrit.models.score import Score
from pyrit.score.true_false.true_false_score_aggregator import AND_, MAJORITY_, OR_


def _mk_score(val: bool, *, prr_id: str, rationale: str = "") -> Score:
    return Score(
        score_value=str(val).lower(),
        score_value_description=str(val),
        score_type="true_false",
        score_category=["test"],
        score_rationale=rationale,
        score_metadata=None,
        prompt_request_response_id=prr_id,
        scorer_class_identifier={"__type__": "UnitTestScorer"},
        objective=None,
    )


def test_and_aggregator_all_true():
    scores = [_mk_score(True, prr_id="1"), _mk_score(True, prr_id="1")]
    res = AND_(scores)
    assert res.value is True
    assert isinstance(res.description, str) and res.description
    assert isinstance(res.rationale, str)


def test_and_aggregator_any_false():
    scores = [_mk_score(True, prr_id="1"), _mk_score(False, prr_id="1")]
    res = AND_(scores)
    assert res.value is False


def test_or_aggregator_any_true():
    scores = [_mk_score(False, prr_id="1"), _mk_score(True, prr_id="1")]
    res = OR_(scores)
    assert res.value is True


def test_or_aggregator_all_false():
    scores = [_mk_score(False, prr_id="1"), _mk_score(False, prr_id="1")]
    res = OR_(scores)
    assert res.value is False


def test_majority_strict_majority_true():
    scores = [
        _mk_score(True, prr_id="1", rationale="A"),
        _mk_score(True, prr_id="1", rationale="B"),
        _mk_score(False, prr_id="1", rationale="C"),
    ]
    res = MAJORITY_(scores)
    assert res.value is True
    assert "MAJORITY" in res.description
    assert "-" in res.rationale or ":" in res.rationale


def test_majority_tie_is_false():
    scores = [_mk_score(True, prr_id="1"), _mk_score(False, prr_id="1")]
    res = MAJORITY_(scores)
    assert res.value is False


def test_category_deduplication():
    """Test that duplicate categories are deduplicated into a single category."""
    scores = [
        Score(
            score_value="true",
            score_value_description="true",
            score_type="true_false",
            score_category=["Hate"],
            score_rationale="test1",
            score_metadata=None,
            prompt_request_response_id="1",
            scorer_class_identifier={"__type__": "UnitTestScorer"},
            objective=None,
        ),
        Score(
            score_value="true",
            score_value_description="true",
            score_type="true_false",
            score_category=["Hate"],
            score_rationale="test2",
            score_metadata=None,
            prompt_request_response_id="1",
            scorer_class_identifier={"__type__": "UnitTestScorer"},
            objective=None,
        ),
    ]
    res = AND_(scores)
    assert res.value is True
    assert res.category == ["Hate"]  # Should be deduplicated to single entry


def test_category_multiple_unique():
    """Test that multiple unique categories are preserved and sorted."""
    scores = [
        Score(
            score_value="true",
            score_value_description="true",
            score_type="true_false",
            score_category=["Violence"],
            score_rationale="test1",
            score_metadata=None,
            prompt_request_response_id="1",
            scorer_class_identifier={"__type__": "UnitTestScorer"},
            objective=None,
        ),
        Score(
            score_value="true",
            score_value_description="true",
            score_type="true_false",
            score_category=["Hate"],
            score_rationale="test2",
            score_metadata=None,
            prompt_request_response_id="1",
            scorer_class_identifier={"__type__": "UnitTestScorer"},
            objective=None,
        ),
    ]
    res = OR_(scores)
    assert res.value is True
    assert res.category == ["Hate", "Violence"]  # Should be sorted alphabetically


def test_category_empty_strings_filtered():
    """Test that empty string categories are filtered out."""
    scores = [
        Score(
            score_value="true",
            score_value_description="true",
            score_type="true_false",
            score_category=[""],
            score_rationale="test1",
            score_metadata=None,
            prompt_request_response_id="1",
            scorer_class_identifier={"__type__": "UnitTestScorer"},
            objective=None,
        ),
        Score(
            score_value="false",
            score_value_description="false",
            score_type="true_false",
            score_category=[""],
            score_rationale="test2",
            score_metadata=None,
            prompt_request_response_id="1",
            scorer_class_identifier={"__type__": "UnitTestScorer"},
            objective=None,
        ),
    ]
    res = MAJORITY_(scores)
    assert res.category == []  # Empty strings should be filtered out


def test_category_mixed_empty_and_valid():
    """Test that empty strings are filtered but valid categories are kept."""
    scores = [
        Score(
            score_value="true",
            score_value_description="true",
            score_type="true_false",
            score_category=[""],
            score_rationale="test1",
            score_metadata=None,
            prompt_request_response_id="1",
            scorer_class_identifier={"__type__": "UnitTestScorer"},
            objective=None,
        ),
        Score(
            score_value="true",
            score_value_description="true",
            score_type="true_false",
            score_category=["Violence"],
            score_rationale="test2",
            score_metadata=None,
            prompt_request_response_id="1",
            scorer_class_identifier={"__type__": "UnitTestScorer"},
            objective=None,
        ),
    ]
    res = AND_(scores)
    assert res.value is True
    assert res.category == ["Violence"]  # Only valid category preserved


def test_category_none_and_empty_list():
    """Test that None and empty list categories result in empty category list."""
    scores = [
        Score(
            score_value="true",
            score_value_description="true",
            score_type="true_false",
            score_category=[],
            score_rationale="test1",
            score_metadata=None,
            prompt_request_response_id="1",
            scorer_class_identifier={"__type__": "UnitTestScorer"},
            objective=None,
        ),
        Score(
            score_value="true",
            score_value_description="true",
            score_type="true_false",
            score_category=[],
            score_rationale="test2",
            score_metadata=None,
            prompt_request_response_id="1",
            scorer_class_identifier={"__type__": "UnitTestScorer"},
            objective=None,
        ),
    ]
    res = OR_(scores)
    assert res.value is True
    assert res.category == []  # Should be empty list
