# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from dataclasses import asdict
from pathlib import Path
from typing import Generator
from unittest.mock import patch

import pytest
from integration.mocks import get_memory_interface

from pyrit.common.path import SCORER_EVALS_HARM_PATH
from pyrit.memory import CentralMemory, MemoryInterface
from pyrit.prompt_target import OpenAIChatTarget
from pyrit.score import (
    HarmScorerEvaluator,
    HarmScorerMetrics,
    LikertScalePaths,
    SelfAskLikertScorer,
)


@pytest.fixture
def memory() -> Generator[MemoryInterface, None, None]:
    yield from get_memory_interface()


@pytest.mark.parametrize(
    "scale_path, dataset_name",
    [
        (LikertScalePaths.HATE_SPEECH_SCALE.value, "hate_speech"),
        (LikertScalePaths.VIOLENCE_SCALE.value, "violence"),
    ],
)
@pytest.mark.asyncio
async def test_likert_scorer_mae(memory, scale_path: Path, dataset_name: str) -> None:
    with patch.object(CentralMemory, "get_memory_instance", return_value=memory):
        target = OpenAIChatTarget(temperature=0.0, seed=1)
        scale_path = LikertScalePaths.HATE_SPEECH_SCALE.value
        likert_scorer = SelfAskLikertScorer(chat_target=target, likert_scale_path=scale_path)

        evaluator = HarmScorerEvaluator(scorer=likert_scorer)
        csv_path = f"{str(SCORER_EVALS_HARM_PATH)}/{dataset_name}.csv"

        baseline_metrics = likert_scorer.get_scorer_metrics(dataset_name=dataset_name)

        assert isinstance(baseline_metrics, HarmScorerMetrics), "Expected HarmScorerMetrics type"
        # using ~95% confidence interval for the mean absolute error (MAE) for thresholding
        mae_threshold = baseline_metrics.mean_absolute_error + (2 * baseline_metrics.mae_standard_error)

        metrics = await evaluator.run_evaluation_from_csv_async(
            csv_path=csv_path,
            assistant_response_col="assistant_response",
            human_label_col_names=["human_likert_score_1", "human_likert_score_2", "human_likert_score_3"],
            objective_or_harm_col_name="category",
            num_scorer_trials=1,
            save_results=False,
        )

        assert (
            metrics.mean_absolute_error <= mae_threshold
        ), f"MAE {metrics.mean_absolute_error} exceeds threshold {mae_threshold}. Full metrics: {asdict(metrics)}"
