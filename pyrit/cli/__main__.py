# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import asyncio
import inspect
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser, Namespace
from datetime import datetime
from pathlib import Path
from typing import List

from pyrit.common import initialize_pyrit
from pyrit.memory import CentralMemory
from pyrit.models.seed_prompt import SeedPromptDataset, SeedPromptGroup

from .scanner_config import ScannerConfig

SCANNER_EXECUTION_START_TIME_MEMORY_LABEL: str = "scanner_execution_start_time"


def parse_args(args=None) -> Namespace:
    parser = ArgumentParser(
        prog="pyrit_scan",
        description="Parse the arguments for the Pyrit Scanner CLI.",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config-file",
        type=str,
        help="Path to the scanner YAML config file.",
        required=True,
    )
    return parser.parse_args(args)


def load_seed_prompt_groups(dataset_paths: List[str]) -> List[SeedPromptGroup]:
    """
    loads each dataset file path into a list of SeedPrompt objects.
    """
    if not dataset_paths:
        raise ValueError("No datasets provided in the configuration.")

    all_prompt_groups: List[SeedPromptGroup] = []
    for path_str in dataset_paths:
        path = Path(path_str)
        if not path.exists():
            raise FileNotFoundError(f"Dataset file '{path}' does not exist.")
        dataset = SeedPromptDataset.from_yaml_file(path)
        groups = SeedPromptDataset.group_seed_prompts_by_prompt_group_id(dataset.prompts)
        all_prompt_groups.extend(groups)
    return all_prompt_groups


def _get_first_text_values_if_exist(prompt_groups: List[SeedPromptGroup]) -> List[str]:
    """
    Get the first text value from the seed prompts in each of the provided prompt groups.

    If no text value exists, return the value of the first seed prompt.

    Args:
        prompt_groups (List[SeedPromptGroup]): List of SeedPromptGroup objects.
            Assumed to contain at least one group, and each group is assumed to
            contain at least one seed prompt.
    """
    first_text_values = []
    for group in prompt_groups:
        if not group.prompts:
            raise ValueError("Seed prompt group is empty, no prompts available.")
        # Find the first text prompt in the group.
        # If none exist, use the first prompt's value.
        first_text_value = group.prompts[0].value
        for prompt in group.prompts:
            if prompt.data_type == "text":
                first_text_value = prompt.value
                break

        first_text_values.append(first_text_value)

    return first_text_values


async def run_scenarios_async(config: ScannerConfig) -> None:
    """
    Run scenarios
    """
    memory_labels = config.database.memory_labels or {}
    memory_labels[SCANNER_EXECUTION_START_TIME_MEMORY_LABEL] = datetime.now().isoformat()

    seed_prompt_groups = load_seed_prompt_groups(config.datasets)
    prompt_converters = config.create_prompt_converters()
    orchestrators = config.create_orchestrators(prompt_converters=prompt_converters)

    for orchestrator in orchestrators:
        objectives = _get_first_text_values_if_exist(seed_prompt_groups)
        if hasattr(orchestrator, "run_attacks_async"):
            args = {
                "objectives": objectives,
                "memory_labels": memory_labels,
            }
            sig = inspect.signature(orchestrator.run_attacks_async)
            if "seed_prompts" in sig.parameters:
                args["seed_prompts"] = seed_prompt_groups
            await orchestrator.run_attacks_async(**args)
        else:
            raise ValueError(f"The orchestrator {type(orchestrator).__name__} does not have run_attacks_async.")

    # Print conversation pieces from memory
    memory = CentralMemory.get_memory_instance()
    all_pieces = memory.get_prompt_request_pieces(labels=memory_labels)
    conversation_id = None
    for piece in all_pieces:
        if piece.conversation_id != conversation_id:
            conversation_id = piece.conversation_id
            print("===================================================")
            print(f"Conversation ID: {conversation_id}")
        print(f"{piece.role}: {piece.converted_value}")


def main(args=None):
    parsed_args = parse_args(args)

    config_file = Path(parsed_args.config_file)
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file {config_file.absolute()} does not exist.")

    config = ScannerConfig.from_yaml(str(config_file))
    initialize_pyrit(memory_db_type=config.database.db_type)

    asyncio.run(run_scenarios_async(config))


if __name__ == "__main__":
    main()
