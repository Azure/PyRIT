# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import asyncio
import inspect
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser, Namespace
from copy import deepcopy
from datetime import datetime
from importlib import import_module
from pathlib import Path
from typing import Any, Dict, List, MutableSequence, Optional
from uuid import uuid4

import yaml

from pyrit.common import initialize_pyrit
from pyrit.memory import CentralMemory
from pyrit.models import SeedPrompt, SeedPromptDataset
from pyrit.models.seed_prompt import SeedPromptGroup
from pyrit.orchestrator import Orchestrator
from pyrit.prompt_converter import PromptConverter
from pyrit.prompt_normalizer.normalizer_request import NormalizerRequest
from pyrit.prompt_normalizer.prompt_converter_configuration import (
    PromptConverterConfiguration,
)
from pyrit.prompt_target import PromptTarget
from pyrit.prompt_target.common.prompt_chat_target import PromptChatTarget
from pyrit.score.scorer import Scorer


def parse_args(args=None) -> Namespace:
    parser = ArgumentParser(
        prog="pyrit_scan",
        description="Parse the arguments for the Pyrit Scanner CLI.",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config-file",
        type=str,
        help="The path to the configuration file.",
        required=True,
    )

    parsed_args = parser.parse_args(args)
    config_file = Path(parsed_args.config_file)
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file {config_file.absolute()} does not exist.")
    return parsed_args


def load_config(config_file: Path) -> Dict[str, Any]:
    # Load the configuration YAML file
    with open(config_file, "r") as file:
        config = yaml.safe_load(file)

    if not config:
        raise ValueError("Configuration file is empty.")

    if not isinstance(config, dict):
        raise TypeError("Configuration file must be a dictionary.")

    return config


async def validate_config_and_run_async(config: Dict[str, Any], memory_labels: Optional[Dict[str, str]] = None) -> None:
    if "scenarios" not in config:
        raise KeyError("Configuration file must contain a 'scenarios' key.")

    scenarios = config["scenarios"]

    if not scenarios:
        raise ValueError("Scenarios list is empty.")

    initialize_pyrit(memory_db_type="DuckDB")

    seed_prompts = generate_datasets(config)
    objective_target = validate_target(config, target_key="objective_target")
    prompt_converters: list[PromptConverter] = []
    # prompt_converters = validate_converters(config)
    adversarial_chat: PromptChatTarget | None = None
    if "adversarial_chat" in config:
        adversarial_chat = validate_target(config, target_key="adversarial_chat")  # type: ignore
    scoring_target = validate_scoring_target(config, adversarial_chat=adversarial_chat)  # type: ignore
    objective_scorer = validate_objective_scorer(config, scoring_target=scoring_target)

    orchestrators = []
    for scenario_config in scenarios:
        orchestrators.append(
            validate_scenario(
                scenario_config=scenario_config,
                objective_target=objective_target,
                adversarial_chat=adversarial_chat,
                prompt_converters=prompt_converters,
                scoring_target=scoring_target,
                objective_scorer=objective_scorer,
            )
        )

    # This is a separate loop because we want to validate all scenarios before starting execution.
    for orchestrator in orchestrators:
        if hasattr(orchestrator, "run_attack_async"):
            for seed_prompt in seed_prompts:
                await orchestrator.run_attack_async(objective=seed_prompt.value, memory_labels=memory_labels)
        elif hasattr(orchestrator, "send_normalizer_requests_async"):
            converter_configurations = [
                PromptConverterConfiguration(converters=prompt_converters if prompt_converters else [])
            ]

            normalizer_requests = [
                NormalizerRequest(
                    seed_prompt_group=SeedPromptGroup(prompts=[seed_prompt]),
                    request_converter_configurations=converter_configurations,
                    conversation_id=str(uuid4()),
                )
                for seed_prompt in seed_prompts
            ]
            await orchestrator.send_normalizer_requests_async(
                prompt_request_list=normalizer_requests,
                memory_labels=memory_labels,
            )
        else:
            supported_methods = ["run_attack_async", "send_normalizer_requests_async"]
            raise ValueError(
                f"The orchestrator of type {type(orchestrator).__name__} does not have a compatible "
                f"method to execute its attack. The supported methods are {supported_methods}."
            )


def validate_scenario(
    scenario_config: Dict[str, Any],
    objective_target: PromptTarget,
    adversarial_chat: Optional[PromptChatTarget] = None,
    prompt_converters: Optional[List[PromptConverter]] = None,
    scoring_target: Optional[PromptChatTarget] = None,
    objective_scorer: Optional[Scorer] = None,
) -> Orchestrator:
    if "type" not in scenario_config:
        raise KeyError("Scenario must contain a 'type' key.")

    scenario_type = scenario_config["type"]
    scenario_args = deepcopy(scenario_config)
    del scenario_args["type"]

    try:
        orchestrator_module = import_module("pyrit.orchestrator")
        orchestrator_class = getattr(orchestrator_module, scenario_type)
    except Exception as ex:
        raise RuntimeError(f"Failed to import orchestrator {scenario_type} from pyrit.orchestrator") from ex

    try:
        constructor_arg_names = [arg.name for arg in inspect.signature(orchestrator_class.__init__).parameters.values()]

        # Some orchestrator arguments have their own configuration since they
        # are more complex. They are passed in as args to this function.
        complex_arg_names = [
            "objective_target",
            "adversarial_chat",
            "prompt_converters",
            "scoring_target",
            "objective_scorer",
        ]
        for complex_arg_name in complex_arg_names:
            if complex_arg_name in scenario_args:
                raise ValueError(
                    f"{complex_arg_name} needs to be configured at the top level of the scanner configuration."
                    f"The scenario configuration cannot include {complex_arg_name}."
                )

            # Add complex args to the argument list.
            local_vars = locals()
            if complex_arg_name in constructor_arg_names:
                arg_value = local_vars[complex_arg_name]
                if arg_value:
                    scenario_args[complex_arg_name] = arg_value

        orchestrator = orchestrator_class(**scenario_args)
    except Exception as ex:
        raise ValueError(f"Failed to validate scenario {scenario_type}: {ex}") from ex
    return orchestrator


def generate_datasets(config: Dict[str, Any]) -> MutableSequence[SeedPrompt]:
    datasets = config.get("datasets")

    if not datasets:
        raise KeyError("Send prompts scenario must contain a 'datasets' key.")

    loaded_dataset_prompts: MutableSequence[SeedPrompt] = []
    for dataset_path in datasets:
        dataset = SeedPromptDataset.from_yaml_file(dataset_path)
        loaded_dataset_prompts.extend(dataset.prompts)

    return loaded_dataset_prompts


def validate_target(config: Dict[str, Any], target_key: str) -> PromptTarget:
    if target_key not in config:
        raise KeyError(f"Configuration file must contain a '{target_key}' key.")

    if not config[target_key] or not config[target_key].get("type"):
        raise KeyError(f"Target {target_key} must contain a 'type' key.")

    target_config = deepcopy(config[target_key])
    target_type = target_config.pop("type")

    try:
        target_module = import_module("pyrit.prompt_target")
        target_class = getattr(target_module, target_type)
    except Exception as ex:
        raise RuntimeError(f"Failed to import target {target_type} from pyrit.prompt_target") from ex

    target = target_class(**target_config)
    return target


def validate_scoring_target(
    config: Dict[str, Any], adversarial_chat: Optional[PromptChatTarget]
) -> PromptChatTarget | None:
    if "scoring" not in config:
        return None
    scoring_config = config["scoring"]

    # If a scoring_target has been configured use it.
    # Otherwise, use the adversarial_chat target for scoring.
    if "scoring_target" in scoring_config:
        return validate_target(scoring_config, target_key="scoring_target")  # type: ignore
    return adversarial_chat


def validate_objective_scorer(config: Dict[str, Any], scoring_target: Optional[PromptChatTarget]) -> Scorer | None:
    if "scoring" not in config:
        return None
    scoring_config = config["scoring"]
    if "objective_scorer" not in scoring_config:
        return None

    scorer_args = deepcopy(scoring_config["objective_scorer"])

    if "type" not in scorer_args:
        raise KeyError("Scorer definition must contain a 'type' key.")

    scorer_type = scorer_args.pop("type")

    try:
        scorer_module = import_module("pyrit.score")
        scorer_class = getattr(scorer_module, scorer_type)
    except Exception as ex:
        raise RuntimeError(f"Failed to import target {scorer_type} from pyrit.score") from ex

    if scoring_target and "chat_target" in inspect.signature(scorer_class.__init__).parameters:
        scorer_args["chat_target"] = scoring_target

    return scorer_class(**scorer_args)


def main(args=None):
    parsed_args = parse_args(args)
    config_file = parsed_args.config_file
    config = load_config(config_file)
    memory_labels = config.get("memory_labels", {})
    # Add timestamp to distinguish between scanner runs with the same memory labels
    memory_labels["scanner_execution_start_time"] = datetime.now().isoformat()

    asyncio.run(validate_config_and_run_async(config, memory_labels))

    memory = CentralMemory.get_memory_instance()
    all_pieces = memory.get_prompt_request_pieces(labels=memory_labels)
    conversation_id = None
    for piece in all_pieces:
        if piece.conversation_id != conversation_id:
            conversation_id = piece.conversation_id
            print("===================================================")
            print(f"Conversation ID: {conversation_id}")
        print(f"{piece.role}: {piece.converted_value}")
