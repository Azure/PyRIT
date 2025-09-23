import uuid

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
