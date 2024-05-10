import time
import numpy as np
import torch.multiprocessing as mp
from ml_collections import config_dict
import pyrit.adv_suffix.llm_attacks.gcg as attack_lib
from pyrit.adv_suffix.llm_attacks.base.attack_manager import get_goals_and_targets, get_workers


class GreedyCoordinateGradientAdversarialSuffixGenerator:
    def __init__(self):
        if mp.get_start_method(allow_none=True) != 'spawn':
            mp.set_start_method("spawn")


    def generate_suffix(
        self,
        token: str = "",
        tokenizer_paths: list = [],
        model_paths: list = [],
        conversation_templates: list = [],
        result_prefix: str = "",
        train_data: str = "",
        control_init: str = "! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! !",
        n_train_data: int = 50,
        n_steps: int = 500,
        test_steps: int = 50,
        data_offset: int = 0,
        batch_size: int = 512,
        transfer: bool = False,
        target_weight: float = 1.0,
        control_weight: float = 0.0,
        progressive_goals: bool = False,
        progressive_models: bool = False,
        anneal: bool = False,
        incr_control: bool = False,
        stop_on_success: bool = False,
        verbose: bool = True,
        allow_non_ascii: bool = False,
        num_train_models: int = 1,
        devices: list = ["cuda:0"],
        model_kwargs: list = [{"low_cpu_mem_usage": True, "use_cache": False}],
        tokenizer_kwargs: list = [{"use_fast": False}],
        n_test_data: int = 0,
        test_data: str = "",
        attack: str = "gcg",
        lr: float = 0.01,
        topk: int = 256,
        temp: int = 1,
        filter_cand: bool = True,
        gbda_deterministic: bool = True
    ):

        params = config_dict.ConfigDict()
        params.result_prefix = result_prefix
        params.train_data = train_data
        params.control_init = control_init
        params.n_train_data = n_train_data
        params.n_steps = n_steps
        params.test_steps = test_steps
        params.data_offset = data_offset
        params.batch_size = batch_size
        params.transfer = transfer
        params.target_weight = target_weight
        params.control_weight = control_weight
        params.progressive_goals = progressive_goals
        params.progressive_models = progressive_models
        params.anneal = anneal
        params.incr_control = incr_control
        params.stop_on_success = stop_on_success
        params.verbose = verbose
        params.allow_non_ascii = allow_non_ascii
        params.num_train_models = num_train_models
        params.tokenizer_paths = tokenizer_paths
        params.tokenizer_kwargs = tokenizer_kwargs
        params.model_paths = model_paths
        params.model_kwargs = model_kwargs
        params.conversation_templates = conversation_templates
        params.devices = devices
        params.n_test_data = n_test_data
        params.test_data = test_data
        params.attack = attack
        params.lr = lr
        params.topk = topk
        params.temp = temp
        params.filter_cand = filter_cand
        params.gbda_deterministic = gbda_deterministic
        params.token = token

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

