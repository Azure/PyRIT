# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from dataclasses import asdict
from typing import Generator
from unittest.mock import patch

import pytest
from integration.mocks import get_memory_interface

from pyrit.common.path import SCORER_EVALS_OBJECTIVE_PATH
from pyrit.memory import CentralMemory
from pyrit.memory.memory_interface import MemoryInterface
from pyrit.prompt_target import OpenAIChatTarget
from pyrit.score import (
    ObjectiveScorerEvaluator,
    ObjectiveScorerMetrics,
    SelfAskRefusalScorer,
)


@pytest.fixture
def memory() -> Generator[MemoryInterface, None, None]:
    yield from get_memory_interface()


@pytest.mark.asyncio
async def test_refusal_scorer_accuracy(memory: MemoryInterface):
    with patch.object(CentralMemory, "get_memory_instance", return_value=memory):
        target = OpenAIChatTarget(temperature=0.0, seed=1)
        refusal_scorer = SelfAskRefusalScorer(chat_target=target)

        evaluator = ObjectiveScorerEvaluator(scorer=refusal_scorer)
        csv_path = f"{str(SCORER_EVALS_OBJECTIVE_PATH)}/mixed_objective_refusal.csv"

        baseline_metrics = refusal_scorer.get_scorer_metrics(dataset_name="mixed_objective_refusal")
        assert isinstance(baseline_metrics, ObjectiveScorerMetrics), "Expected HarmScorerMetrics type"
        # if 95% confidence interval for the accuracy is higher than 0.95, use 0.95 as the threshold
        accuracy_threshold = min(baseline_metrics.accuracy - (2 * baseline_metrics.accuracy_standard_error), 0.95)

        metrics = await evaluator.run_evaluation_from_csv_async(
            csv_path=csv_path,
            assistant_response_col_name="assistant_message",
            human_label_col_names=["human_score"],
            objective_or_harm_col_name="objective",
            num_scorer_trials=1,
            save_results=False,
        )

        assert (
            metrics.accuracy >= accuracy_threshold
        ), f"Accuracy {metrics.accuracy} is below threshold {accuracy_threshold}. Full metrics: {asdict(metrics)}"
