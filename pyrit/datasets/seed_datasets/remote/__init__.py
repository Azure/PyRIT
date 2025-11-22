# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Remote dataset loaders with automatic discovery.

Import concrete implementations to trigger registration.
"""

from pyrit.datasets.seed_datasets.remote.aya_redteaming_dataset import AyaRedteamingDataset  # noqa: F401
from pyrit.datasets.seed_datasets.remote.babelscape_alert_dataset import BabelscapeAlertDataset  # noqa: F401
from pyrit.datasets.seed_datasets.remote.ccp_sensitive_prompts_dataset import CCPSensitivePromptsDataset  # noqa: F401
from pyrit.datasets.seed_datasets.remote.darkbench_dataset import DarkBenchDataset  # noqa: F401
from pyrit.datasets.seed_datasets.remote.equitymedqa_dataset import EquityMedQADataset  # noqa: F401
from pyrit.datasets.seed_datasets.remote.forbidden_questions_dataset import ForbiddenQuestionsDataset  # noqa: F401
from pyrit.datasets.seed_datasets.remote.harmbench_dataset import HarmBenchDataset  # noqa: F401
from pyrit.datasets.seed_datasets.remote.harmbench_multimodal_dataset import HarmBenchMultimodalDataset  # noqa: F401
from pyrit.datasets.seed_datasets.remote.jbb_behaviors_dataset import JBBBehaviorsDataset  # noqa: F401
from pyrit.datasets.seed_datasets.remote.librai_do_not_answer_dataset import LibrAIDoNotAnswerDataset  # noqa: F401
from pyrit.datasets.seed_datasets.remote.llm_latent_adversarial_training_dataset import LLMLatentAdversarialTrainingDataset  # noqa: F401
from pyrit.datasets.seed_datasets.remote.many_shot_jailbreaking_dataset import ManyShotJailbreakingDataset  # noqa: F401
from pyrit.datasets.seed_datasets.remote.medsafetybench_dataset import MedSafetyBenchDataset  # noqa: F401
from pyrit.datasets.seed_datasets.remote.mlcommons_ailuminate_dataset import MLCommonsAILuminateDataset  # noqa: F401
from pyrit.datasets.seed_datasets.remote.multilingual_vulnerability_dataset import MultilingualVulnerabilityDataset  # noqa: F401
from pyrit.datasets.seed_datasets.remote.pku_safe_rlhf_dataset import PKUSafeRLHFDataset  # noqa: F401
from pyrit.datasets.seed_datasets.remote.red_team_social_bias_dataset import RedTeamSocialBiasDataset  # noqa: F401
from pyrit.datasets.seed_datasets.remote.remote_dataset_loader import RemoteDatasetLoader
from pyrit.datasets.seed_datasets.remote.sorry_bench_dataset import SorryBenchDataset  # noqa: F401
from pyrit.datasets.seed_datasets.remote.sosbench_dataset import SOSBenchDataset  # noqa: F401
from pyrit.datasets.seed_datasets.remote.tdc23_redteaming_dataset import TDC23RedteamingDataset  # noqa: F401
from pyrit.datasets.seed_datasets.remote.xstest_dataset import XSTestDataset  # noqa: F401

__all__ = [
    "RemoteDatasetLoader",
]
