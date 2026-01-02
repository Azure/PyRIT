# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from dataclasses import asdict
from typing import Generator
from unittest.mock import patch

import pytest
from integration.mocks import get_memory_interface

from pyrit.memory import CentralMemory
from pyrit.memory.memory_interface import MemoryInterface
from pyrit.prompt_target import OpenAIChatTarget
from pyrit.score import (
    ObjectiveScorerEvaluator,
    ObjectiveScorerMetrics,
    RegistryUpdateBehavior,
    ScorerEvalDatasetFiles,
    SelfAskRefusalScorer,
)


@pytest.fixture
def memory() -> Generator[MemoryInterface, None, None]:
    yield from get_memory_interface()


@pytest.mark.run_only_if_all_tests
@pytest.mark.asyncio
async def test_refusal_scorer_accuracy(memory: MemoryInterface):
    with patch.object(CentralMemory, "get_memory_instance", return_value=memory):
        target = OpenAIChatTarget(temperature=0.0, seed=1)
        refusal_scorer = SelfAskRefusalScorer(chat_target=target)

        evaluator = ObjectiveScorerEvaluator(scorer=refusal_scorer)

        baseline_metrics = refusal_scorer.get_scorer_metrics()
        assert isinstance(baseline_metrics, ObjectiveScorerMetrics), "Expected ObjectiveScorerMetrics type"
        # if 95% confidence interval for the accuracy is higher than 0.95, use 0.95 as the threshold
        accuracy_threshold = min(baseline_metrics.accuracy - (2 * baseline_metrics.accuracy_standard_error), 0.95)

        dataset_files = ScorerEvalDatasetFiles(
            human_labeled_datasets_files=["mixed_objective_refusal.csv"],
            result_file="test_refusal_metrics.jsonl",
        )

        metrics = await evaluator.run_evaluation_async(
            dataset_files=dataset_files,
            num_scorer_trials=1,
            update_registry_behavior=RegistryUpdateBehavior.NEVER_UPDATE,
        )

        assert metrics is not None, "Evaluation returned None metrics"
        assert isinstance(metrics, ObjectiveScorerMetrics), "Expected ObjectiveScorerMetrics type"
        assert (
            metrics.accuracy >= accuracy_threshold
        ), f"Accuracy {metrics.accuracy} is below threshold {accuracy_threshold}. Full metrics: {asdict(metrics)}"
