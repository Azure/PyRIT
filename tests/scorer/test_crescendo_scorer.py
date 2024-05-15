# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from pyrit.score.scorer_strategy import ScorerStrategy
from pyrit.score.crescendo_scorer import EVALUATOR, META_JUDGE_PROMPT, REFUSE_JUDGE_PROMPT
import json


def test_load_eval_prompt() -> None:
    scorer_strategy = ScorerStrategy(strategy=EVALUATOR, conversation_objective="1", response_to_evaluate_input="2")

    assert scorer_strategy.strategy.name == "Crescendo Eval Prompt"
    assert scorer_strategy.kwargs["conversation_objective"] == "1"
    assert scorer_strategy.kwargs["response_to_evaluate_input"] == "2"
    assert str(scorer_strategy)


def test_load_meta_judge_prompt() -> None:
    scorer_strategy = ScorerStrategy(strategy=META_JUDGE_PROMPT, result_explanation="1")

    assert scorer_strategy.strategy.name == "Crescendo Meta Judge Prompt"
    assert scorer_strategy.kwargs["result_explanation"] == "1"
    assert str(scorer_strategy)


def test_load_refuse_judge_prompt() -> None:
    scorer_strategy = ScorerStrategy(
        strategy=REFUSE_JUDGE_PROMPT, conversation_objective="1", response_to_evaluate_input="2"
    )

    assert scorer_strategy.strategy.name == "Crescendo Refuse Judge Prompt"
    assert scorer_strategy.kwargs["conversation_objective"] == "1"
    assert scorer_strategy.kwargs["response_to_evaluate_input"] == "2"
    assert str(scorer_strategy)


def test_json() -> None:
    kwargs = {}
    kwargs["conversation_objective"] = "1"
    kwargs["response_to_evaluate_input"] = "2"

    # convert kwargs dict to json
    json_kwargs = json.dumps(kwargs)

    # convert back to dict
    kwargs2 = json.loads(json_kwargs)
    scorer_strategy = ScorerStrategy(strategy=REFUSE_JUDGE_PROMPT, **kwargs2)
    assert scorer_strategy.strategy.name == "Crescendo Refuse Judge Prompt"
    assert scorer_strategy.kwargs["conversation_objective"] == "1"
    assert scorer_strategy.kwargs["response_to_evaluate_input"] == "2"
    assert str(scorer_strategy)
