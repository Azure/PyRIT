import os
import yaml
from pyrit.common import default_values
from train import GreedyCoordinateGradientAdversarialSuffixGenerator

def _load_yaml_to_dict(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        data = yaml.safe_load(f)
    return data


def run_trainer(
    model_name: str,
    setup: str = "single",
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

    if model_name not in ["mistral", "llama_2", "llama_3", "llama_mistral", "vicuna"]:
        raise ValueError("Model name not supported. Currently supports 'mistral' and 'llama2'")
    
    default_values.load_default_env()
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        raise ValueError("Please set the HF_TOKEN environment variable")
    runtime_config = {
        "train_data": f"https://raw.githubusercontent.com/llm-attacks/llm-attacks/main/data/advbench/harmful_behaviors.csv",
        "result_prefix": f"results/individual_behaviors_{model_name}_gcg_offset_{data_offset}",
        "token": hf_token
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
    if not os.path.exists('results'):
        os.makedirs('results')

    trainer.generate_suffix(**config)


if __name__ == '__main__':
    run_trainer(model_name = "vicuna", setup = "single", n_train_data = 1, n_steps = 150, test_steps = 25, batch_size = 256)
    # run_trainer(model_name = "mistral", setup = "multiple", n_train_data = 10, n_test_data=3, n_steps = 40, test_steps = 1, batch_size = 128)
    # run_trainer(model_name = "llama_mistral", setup = "multiple", num_train_models = 2, n_train_data = 2, n_test_data=1, n_steps = 10, test_steps = 1, batch_size = 64)



