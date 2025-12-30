# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from dataclasses import asdict
from typing import Generator
from unittest.mock import patch

import pytest
from integration.mocks import get_memory_interface

from pyrit.memory import CentralMemory, MemoryInterface
from pyrit.prompt_target import OpenAIChatTarget
from pyrit.score import (
    HarmScorerEvaluator,
    HarmScorerMetrics,
    LikertScalePaths,
    RegistryUpdateBehavior,
    ScorerEvalDatasetFiles,
    SelfAskLikertScorer,
)


@pytest.fixture
def memory() -> Generator[MemoryInterface, None, None]:
    yield from get_memory_interface()


@pytest.mark.run_only_if_all_tests
@pytest.mark.parametrize(
    "likert_scale, dataset_name",
    [
        (LikertScalePaths.HATE_SPEECH_SCALE, "hate_speech"),
        (LikertScalePaths.VIOLENCE_SCALE, "violence"),
    ],
)
@pytest.mark.asyncio
async def test_likert_scorer_mae(memory, likert_scale: LikertScalePaths, dataset_name: str) -> None:
    with patch.object(CentralMemory, "get_memory_instance", return_value=memory):
        target = OpenAIChatTarget(temperature=0.0, seed=1)
        likert_scorer = SelfAskLikertScorer(chat_target=target, likert_scale=likert_scale)

        evaluator = HarmScorerEvaluator(scorer=likert_scorer)

        baseline_metrics = likert_scorer.get_scorer_metrics()

        assert isinstance(baseline_metrics, HarmScorerMetrics), "Expected HarmScorerMetrics type"
        # using ~95% confidence interval for the mean absolute error (MAE) for thresholding
        mae_threshold = baseline_metrics.mean_absolute_error + (2 * baseline_metrics.mae_standard_error)

        dataset_files = ScorerEvalDatasetFiles(
            human_labeled_datasets_files=[f"{dataset_name}.csv"],
            result_file=f"test_{dataset_name}_metrics.jsonl",
            harm_category=dataset_name,
        )

        metrics = await evaluator.run_evaluation_async(
            dataset_files=dataset_files,
            num_scorer_trials=1,
            update_registry_behavior=RegistryUpdateBehavior.NEVER_UPDATE,
        )

        assert metrics is not None, "Evaluation returned None metrics"
        assert isinstance(metrics, HarmScorerMetrics), "Expected HarmScorerMetrics type"
        assert (
            metrics.mean_absolute_error <= mae_threshold
        ), f"MAE {metrics.mean_absolute_error} exceeds threshold {mae_threshold}. Full metrics: {asdict(metrics)}"
