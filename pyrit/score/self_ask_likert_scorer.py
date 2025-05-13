# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import enum
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import krippendorff
import numpy as np
import yaml
from scipy.stats import ttest_1samp

from pyrit.common.path import LIKERT_SCALES_PATH, SCORER_EVALS_PATH
from pyrit.common.scorer_evals_helper import (
    extract_all_human_scores,
    load_data_and_validate_cols,
    save_metrics_json,
    save_scores_to_csv,
)
from pyrit.models import PromptRequestPiece, Score, SeedPrompt, UnvalidatedScore
from pyrit.prompt_target import PromptChatTarget
from pyrit.score.scorer import Scorer, ScorerEvalConfig

logger = logging.getLogger(__name__)


class LikertScalePaths(enum.Enum):
    CYBER_SCALE = Path(LIKERT_SCALES_PATH, "cyber.yaml").resolve()
    FAIRNESS_BIAS_SCALE = Path(LIKERT_SCALES_PATH, "fairness_bias.yaml").resolve()
    HARM_SCALE = Path(LIKERT_SCALES_PATH, "harm.yaml").resolve()
    HATE_SPEECH_SCALE = Path(LIKERT_SCALES_PATH, "hate_speech.yaml").resolve()
    MISINFORMATION_SCALE = Path(LIKERT_SCALES_PATH, "misinformation.yaml").resolve()
    PERSUASION_SCALE = Path(LIKERT_SCALES_PATH, "persuasion.yaml").resolve()
    PHISHING_SCALE = Path(LIKERT_SCALES_PATH, "phishing.yaml").resolve()
    SEXUAL_SCALE = Path(LIKERT_SCALES_PATH, "sexual.yaml").resolve()
    VIOLENCE_SCALE = Path(LIKERT_SCALES_PATH, "violence.yaml").resolve()
    DECEPTION_SCALE = Path(LIKERT_SCALES_PATH, "deception.yaml").resolve()


class SelfAskLikertScorer(Scorer):
    """
    A class that represents a "self-ask" score for text scoring for a likert scale.
    """

    EVALS_DIR_PATH = Path(SCORER_EVALS_PATH, "metrics", "likert_scorer").resolve()

    def __init__(self, chat_target: PromptChatTarget, likert_scale_path: Path) -> None:
        self._prompt_target = chat_target
        self.scorer_type = "float_scale"

        likert_scale = yaml.safe_load(likert_scale_path.read_text(encoding="utf-8"))

        if likert_scale["category"]:
            self._score_category = likert_scale["category"]
        else:
            raise ValueError(f"Improperly formatted likert scale yaml file. Missing category in {likert_scale_path}.")

        likert_scale = self._likert_scale_description_to_string(likert_scale["scale_descriptions"])

        scoring_instructions_template = SeedPrompt.from_yaml_file(LIKERT_SCALES_PATH / "likert_system_prompt.yaml")
        self._system_prompt = scoring_instructions_template.render_template_value(
            likert_scale=likert_scale, category=self._score_category
        )

    def _likert_scale_description_to_string(self, descriptions: list[Dict[str, str]]) -> str:
        """
        Converts the Likert scales to a string representation to be put in a system prompt.

        Args:
            descriptions: list[Dict[str, str]]: The Likert scale to use.

        Returns:
            str: The string representation of the Likert scale.
        """
        if not descriptions:
            raise ValueError("Impropoerly formated Likert scale yaml file. No likert scale_descriptions provided")

        likert_scale_description = ""

        for description in descriptions:
            name = description["score_value"]
            desc = description["description"]

            if int(name) < 0 or int(name) > 5:
                raise ValueError(
                    "Impropoerly formated Likert scale yaml file. Likert scale values must be between 1 and 5"
                )

            likert_scale_description += f"'{name}': {desc}\n"

        return likert_scale_description

    async def score_async(self, request_response: PromptRequestPiece, *, task: Optional[str] = None) -> list[Score]:
        """
        Scores the given request_response using "self-ask" for the chat target and adds score to memory.

        Args:
            request_response (PromptRequestPiece): The prompt request piece containing the text to be scored.
            task (str): The task based on which the text should be scored (the original attacker model's objective).
                Currently not supported for this scorer.

        Returns:
            list[Score]: The request_response scored.
                         The category is configured from the likert_scale.
                         The score_value is a value from [0,1] that is scaled from the likert scale.
        """
        self.validate(request_response, task=task)

        unvalidated_score: UnvalidatedScore = await self._score_value_with_llm(
            prompt_target=self._prompt_target,
            system_prompt=self._system_prompt,
            prompt_request_value=request_response.converted_value,
            prompt_request_data_type=request_response.converted_value_data_type,
            scored_prompt_id=request_response.id,
            category=self._score_category,
            task=task,
        )

        score = unvalidated_score.to_score(
            score_value=str(self.scale_value_float(float(unvalidated_score.raw_score_value), 1, 5)),
        )

        score.score_metadata = str({"likert_value": str(unvalidated_score.raw_score_value)})

        self._memory.add_scores_to_memory(scores=[score])
        return [score]

    def validate(self, request_response: PromptRequestPiece, *, task: Optional[str] = None):
        pass

    async def run_evaluation(self, config: ScorerEvalConfig, batch_size: int = 10) -> str:
        """
        Run evaluation for the scorer using the provided configuration. The metrics calculated for this scorer include:
        Mean Absolute Error (MAE), standard error of the MAE, t-statistic and p-value (from 1-sample t-test using
        model scores - human scores), and Krippendorff's alpha for inter-rater reliability across all model and
        human scores.

        Args:
            config (ScorerEvalConfig): The configuration for the evaluation.

        Returns:
            str: The evaluation results in JSON format.
        """
        assistant_responses, all_human_scores, avg_human_scores = self._prepare_eval_data(config=config)

        all_model_scores, avg_model_scores = await self._run_model_trials_async(
            responses=assistant_responses, num_trials=config.scorer_trials, batch_size=batch_size
        )

        timestamp = datetime.now().strftime("%Y_%m_%d-%H_%M")
        file_name = f"{self.__class__.__name__}_eval_{timestamp}"
        if config.csv_scores_save_dir:
            save_scores_to_csv(
                config=config,
                responses=assistant_responses,
                all_model_scores=all_model_scores,
                avg_model_scores=avg_model_scores,
                avg_human_scores=avg_human_scores,
                file_name=file_name,
            )

        reliability_data = np.concatenate((all_human_scores, all_model_scores))
        eval_dict = self._compute_eval_metrics(avg_human_scores, avg_model_scores, reliability_data)

        if config.json_output_save_dir:
            save_metrics_json(config=config, eval_dict=eval_dict, file_name=file_name)

        return json.dumps(eval_dict)

    def _prepare_eval_data(self, config: ScorerEvalConfig):
        if config.tasks_col_name:
            logger.warning("Task-based scoring is not supported for this scorer. Task column will be ignored.")
        eval_df = load_data_and_validate_cols(config=config, tasks_accepted=False)
        assistant_responses = eval_df[config.assistant_response_col_name].to_list()
        all_human_scores = extract_all_human_scores(eval_df, config)
        if not config.normalized:
            all_human_scores = (all_human_scores - 1) / 4
        average_human_scores = np.mean(all_human_scores, axis=0)
        return assistant_responses, all_human_scores, average_human_scores

    async def _run_model_trials_async(self, responses: list[str], num_trials: int, batch_size: int):
        all_model_scores_list = []
        for trial in range(num_trials):
            scores = await self.score_text_batch_async(texts=responses, tasks=None, batch_size=batch_size)
            score_values = [score.get_value() for score in scores]
            assert len(score_values) == len(
                responses
            ), f"Expected {len(responses)} scores in trial {trial} but got {len(score_values)}"
            assert all(
                isinstance(value, float) for value in score_values
            ), f"Expected all scores to be float but got {type(score_values[0])}"
            all_model_scores_list.append(score_values)
        all_model_scores = np.array(all_model_scores_list)
        average_model_scores = np.mean(all_model_scores, axis=0)
        return all_model_scores, average_model_scores

    def _compute_eval_metrics(self, avg_human_scores, avg_model_scores, all_scores_combined):
        diff = avg_model_scores - avg_human_scores
        abs_error = np.abs(diff)
        eval_dict = {
            "MAE": np.mean(abs_error),
            "MAE_std_error": np.std(abs_error) / np.sqrt(len(abs_error)),
            # t-statistic and p-value via 1-sample t-test for the null hypothesis that the mean difference is 0
            # High positive t-statistic and low p-value indicate model scores are typically higher than human scores
            "t-statistic": ttest_1samp(diff, 0).statistic,
            "p-value": ttest_1samp(diff, 0).pvalue,
            "Krippendorff Alpha": krippendorff.alpha(
                reliability_data=all_scores_combined, value_domain=np.linspace(0, 1, 5), level_of_measurement="ordinal"
            ),
        }
        return eval_dict
