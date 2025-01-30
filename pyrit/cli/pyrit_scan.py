# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import argparse
import asyncio
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List
import yaml

from pyrit.common import initialize_pyrit
from pyrit.memory import CentralMemory
from pyrit.models import SeedPromptDataset
from pyrit.orchestrator import PromptSendingOrchestrator
from pyrit.prompt_target import AzureMLChatTarget, OpenAIChatTarget, PromptTarget


class Scenario:
    async def run():
        pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Parse the arguments for the Pyrit Scanner CLI.")
    parser.add_argument(
        "--config-file",
        type=str,
        help="The path to the configuration file.",
    )

    args = parser.parse_args()
    config_file = Path(args.config_file)
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file {config_file} does not exist.")
    return args


def load_config(config_file: Path) -> Dict[str, Any]:
    # Load the configuration YAML file
    with open(config_file, "r") as file:
        config = yaml.safe_load(file)
    
    if not config:
        raise ValueError("Configuration file is empty.")
    
    if not isinstance(config, dict):
        raise TypeError("Configuration file must be a dictionary.")

    return config


async def validate_config_and_run(config: Dict[str, Any]) -> None:    
    if "scenarios" not in config:
        raise KeyError("Configuration file must contain a 'scenarios' key.")
    
    scenarios = config["scenarios"]

    if not scenarios:
        raise ValueError("Scenarios list is empty.")

    initialize_pyrit(memory_db_type="DuckDB")

    for scenario in scenarios:
        scenario = validate_scenario(scenario, config)
        await scenario.run()


def validate_scenario(scenario: Dict[str, Any], config: Dict[str, Any]) -> Scenario:
    if "type" not in scenario:
        raise KeyError("Scenario must contain a 'type' key.")
    
    scenario_type = scenario["type"]
    if scenario_type == "send_prompts":
        return validate_send_prompts(scenario, config)
    else:
        raise ValueError(f"Invalid scenario type: {scenario_type}")


def validate_datasets(config: Dict[str, Any]) -> List[SeedPromptDataset]:
    datasets = config.get("datasets")

    if not datasets:
        raise KeyError("Send prompts scenario must contain a 'datasets' key.")
    
    loaded_datasets = []
    for dataset_path in datasets:
        if not Path(dataset_path).exists():
            raise FileNotFoundError(f"Dataset file {dataset_path} does not exist.")
        
        loaded_datasets.append(SeedPromptDataset.from_yaml_file(dataset_path))
        
    return loaded_datasets


def validate_objective_target(config: Dict[str, Any]) -> PromptTarget:
    if "objective_target" not in config:
        raise KeyError("Configuration file must contain an 'objective_target' key.")
    
    objective_target_type = config["objective_target"].get("type")

    if not objective_target_type:
        raise KeyError("Objective target must contain a 'type' key.")

    type_to_class_map = {
        "azure_ml": AzureMLChatTarget,
        "openai": OpenAIChatTarget,
    }

    if objective_target_type not in type_to_class_map:
        raise ValueError(f"Invalid objective target type: {objective_target_type}. Must be one of {type_to_class_map.keys()}.")
    
    objective_target_class = type_to_class_map[objective_target_type]
    objective_target_config = deepcopy(config["objective_target"])
    # type is not an actual arg so remove it
    del objective_target_config["type"]
    objective_target = objective_target_class(**objective_target_config)
    return objective_target


def validate_send_prompts(scenario: Dict[str, Any], config: Dict[str, Any]) -> Scenario:
    objective_target = validate_objective_target(config)
    # converters = validate_converters(config)
    # scorers = validate_scorers(config)

    orchestrator = PromptSendingOrchestrator(
        objective_target=objective_target,
        prompt_converters=[],
        scorers=None,
    )

    datasets = validate_datasets(config)

    consolidated_prompt_list = [prompt.value for dataset in datasets for prompt in dataset.prompts]

    class PromptSendingScenario(Scenario):
        async def run(self):
            await orchestrator.send_prompts_async(
                prompt_list=consolidated_prompt_list,
                memory_labels=config.get("memory_labels"),
            )
    
    return PromptSendingScenario()
        


if __name__ == "__main__":
    args = parse_args()
    config_file = args.config_file
    config = load_config(config_file)
    memory_labels = config.get("memory_labels", {})
    # Add timestamp to distinguish between scanner runs with the same memory labels
    memory_labels["scanner_execution_start_time"] = datetime.now().isoformat()

    asyncio.run(validate_config_and_run(config))

    memory = CentralMemory.get_memory_instance()
    all_pieces = memory.get_prompt_request_pieces(labels=memory_labels)
    conversation_id = None
    for piece in all_pieces:
        if piece.conversation_id != conversation_id:
            conversation_id = piece.conversation_id
            print("===================================================")
            print(f"Conversation ID: {conversation_id}")
        print(f"{piece.role}: {piece.converted_value}")