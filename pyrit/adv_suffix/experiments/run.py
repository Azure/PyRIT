import os
import yaml
from train import GreedyCoordinateGradientAdversarialSuffixGenerator
from pyrit.common import default_values

def _load_yaml_to_dict(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        data = yaml.safe_load(f)
    return data


def run_trainer(
    model_name: str,
    setup: str = "behaviors",
    data_offset: int = 0,
    **extra_config_parameters
):
    """
    Trains and generates adversarial suffix - single model single prompt

    Args:
        model_name (str): The name of the model, currently supports "mistral" and "llama2"
        setup (str): Identifier for the setup, currently supporst "behavors"
        data_offset (int): Offset index to start from in the training data. Default is 0.
    """

    if model_name not in ["mistral", "llama2"]:
        raise ValueError("Model name not supported. Currently supports 'mistral' and 'llama2'")
    
    default_values.load_default_env()
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        raise ValueError("Please set the HF_TOKEN environment variable")

    config = {
        "train_data_path":  f"https://raw.githubusercontent.com/llm-attacks/llm-attacks/main/data/advbench/harmful_{setup}.csv",
        "result_prefix": f"results/individual_{setup}_{model_name}_gcg_offset_{data_offset}",
        "token": hf_token
    }
    if model_name == "mistral":
        config = _load_yaml_to_dict("configs/individual_mistral.yaml")
    if model_name == "llama2":
        config = _load_yaml_to_dict("configs/individual_llama_2.yaml")


    config.update(extra_config_parameters)

    trainer = GreedyCoordinateGradientAdversarialSuffixGenerator()
    if not os.path.exists('results'):
        os.makedirs('results')

    trainer.generate_suffix(**config)


if __name__ == '__main__':
    run_trainer(model_name = "mistral", setup = "behaviors")

