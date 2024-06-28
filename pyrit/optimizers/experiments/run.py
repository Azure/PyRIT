import os
import yaml
from typing import Union, Dict, Any
from pyrit.common import default_values
from train import GreedyCoordinateGradientAdversarialSuffixGenerator


def _load_yaml_to_dict(config_path: str) -> dict:
    with open(config_path, "r") as f:
        data = yaml.safe_load(f)
    return data


def run_trainer(model_name: str, setup: str = "single", **extra_config_parameters):
    """
    Trains and generates adversarial suffix - single model single prompt

    Args:
        model_name (str): The name of the model, currently supports:
            "mistral", "llama_2", "llama_3", "vicuna", "all_models"
        setup (str): Identifier for the setup, currently supporst
            - "single": one prompt one model
            - "multiple": multiple prompts one model or multiple prompts multiple models

    """

    if model_name not in ["mistral", "llama_2", "llama_3", "vicuna", "all_models"]:
        raise ValueError("Model name not supported. Currently supports 'mistral' and 'llama2'")

    default_values.load_default_env()
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        raise ValueError("Please set the HF_TOKEN environment variable")
    runtime_config: Dict[str, Union[str, bool, Any]] = {
        "train_data": (
            "https://raw.githubusercontent.com/llm-attacks/llm-attacks/main/data/advbench/harmful_behaviors.csv"
        ),
        "result_prefix": f"results/individual_behaviors_{model_name}_gcg",
        "token": hf_token,
    }
    if setup != "single":
        runtime_config["progressive_goals"] = True
        runtime_config["stop_on_success"] = True
        config_name = "transfer"
    else:
        config_name = "individual"

    config = _load_yaml_to_dict(f"configs/{config_name}_{model_name}.yaml")

    config.update(runtime_config)
    config.update(extra_config_parameters)

    trainer = GreedyCoordinateGradientAdversarialSuffixGenerator()
    if not os.path.exists("results"):
        os.makedirs("results")

    trainer.generate_suffix(**config)


if __name__ == "__main__":
    run_trainer(model_name="vicuna", setup="single", n_train_data=1, n_steps=150, test_steps=25, batch_size=256)

    run_trainer(model_name="mistral", setup="multiple", n_train_data=2, n_steps=40, test_steps=1, batch_size=128)

    run_trainer(
        model_name="all_models",
        setup="multiple",
        num_train_models=4,
        n_train_data=25,
        n_steps=100,
        test_steps=1,
        batch_size=512,
        random_seed=42,
    )
