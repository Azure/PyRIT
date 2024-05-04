from absl import flags
from ml_collections import config_flags
import time
import numpy as np
import sys
import os
import torch.multiprocessing as mp

base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(base_path)
from llm_attacks.base.attack_manager import get_goals_and_targets, get_workers
import llm_attacks.gcg as attack_lib


class GreedyCoordinateGradientAdversarialSuffixGenerator:
    def __init__(self, config_path):
        if mp.get_start_method(allow_none=True) != 'spawn':
            mp.set_start_method("spawn")
        self._CONFIG = config_flags.DEFINE_config_file("config")
        flags.FLAGS(['main.py', '--config', config_path])


    def generate_suffix(self, train_data, result_prefix, n_train_data, n_steps, test_steps, data_offset, batch_size):

        params = self._CONFIG.value
        params.train_data = train_data
        params.result_prefix = result_prefix
        params.n_train_data = n_train_data
        params.n_steps = n_steps
        params.test_steps = test_steps
        params.data_offset = data_offset
        params.batch_size = batch_size

        train_goals, train_targets, test_goals, test_targets = get_goals_and_targets(params)
        process_fn = lambda s: s.replace("Sure, h", "H")
        process_fn2 = lambda s: s.replace("Sure, here is", "Sure, here's")
        train_targets = [process_fn(t) if np.random.random() < 0.5 else process_fn2(t) for t in train_targets]
        test_targets = [process_fn(t) if np.random.random() < 0.5 else process_fn2(t) for t in test_targets]

        workers, test_workers = get_workers(params)
        managers = {
            "AP": attack_lib.AttackPrompt,
            "PM": attack_lib.PromptManager,
            "MPA": attack_lib.MultiPromptAttack,
        }

        timestamp = time.strftime("%Y%m%d-%H:%M:%S")
        if params.transfer:
            attack = attack_lib.ProgressiveMultiPromptAttack(
                train_goals,
                train_targets,
                workers,
                progressive_models=params.progressive_models,
                progressive_goals=params.progressive_goals,
                control_init=params.control_init,
                logfile=f"{params.result_prefix}_{timestamp}.json",
                managers=managers,
                test_goals=test_goals,
                test_targets=test_targets,
                test_workers=test_workers,
                mpa_deterministic=params.gbda_deterministic,
                mpa_lr=params.lr,
                mpa_batch_size=params.batch_size,
                mpa_n_steps=params.n_steps,
            )
        else:
            print("attack")
            attack = attack_lib.IndividualPromptAttack(
                train_goals,
                train_targets,
                workers,
                control_init=params.control_init,
                logfile=f"{params.result_prefix}_{timestamp}.json",
                managers=managers,
                test_goals=getattr(params, "test_goals", []),
                test_targets=getattr(params, "test_targets", []),
                test_workers=test_workers,
                mpa_deterministic=params.gbda_deterministic,
                mpa_lr=params.lr,
                mpa_batch_size=params.batch_size,
                mpa_n_steps=params.n_steps,
            )
        attack.run(
            n_steps=params.n_steps,
            batch_size=params.batch_size,
            topk=params.topk,
            temp=params.temp,
            target_weight=params.target_weight,
            control_weight=params.control_weight,
            test_steps=getattr(params, "test_steps", 1),
            anneal=params.anneal,
            incr_control=params.incr_control,
            stop_on_success=params.stop_on_success,
            verbose=params.verbose,
            filter_cand=params.filter_cand,
            allow_non_ascii=params.allow_non_ascii,
        )

        for worker in workers + test_workers:
            worker.stop()

