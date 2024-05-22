import os
import yaml
from pyrit.common import default_values
from pyrit.optimizers.experiments.eval import AdversarialSuffixEvaluator


def _load_yaml_to_dict(config_path: str) -> dict:
    with open(config_path, "r") as f:
        data = yaml.safe_load(f)
    return data


def run_evaluator(model_name: str, logdir="results/", **extra_config_parameters):
    """
    Evaluate suffixes generated - single model single prompt

    Args:
        model_name (str): The name of the model, should be consistent with the name used in suffix genearation
        batch_size (int): Size of the batch used in training. Default is 128.
    """

    if model_name not in ["mistral", "llama2"]:
        raise ValueError("Model name not supported. Currently supports 'mistral' and 'llama2'")

    default_values.load_default_env()
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        raise ValueError("Please set the HF_TOKEN environment variable")

    runtime_config = {
        "train_data": (
            "https://raw.githubusercontent.com/llm-attacks/llm-attacks/main/data/advbench/harmful_behaviors.csv"
        ),
        "logfile": "results/individual_behavior_controls.json",
        "token": hf_token,
    }
    if model_name == "mistral":
        config = _load_yaml_to_dict("configs/transfer_mistral.yaml")
    if model_name == "llama2":
        config = _load_yaml_to_dict("configs/transfer_llama_2.yaml")

    config.update(runtime_config)
    config.update(extra_config_parameters)

    evaluator = AdversarialSuffixEvaluator(logdir, setup)
    evaluator.parse_logs()

    if not os.path.exists("eval"):
        os.makedirs("eval")

    evaluator.evaluate_suffix(model_name=model_name, **config)


if __name__ == "__main__":
    run_evaluator(model_name="mistral")
