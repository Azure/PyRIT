from test_main import GreedyCoordinateGradientAdversarialSuffixGenerator
import torch.multiprocessing as mp

def run_trainer(model_name, n_train_data, n_steps, test_steps=25):
    if mp.get_start_method(allow_none=True) != 'spawn':
        mp.set_start_method("spawn")

    config_path = f"configs/individual_{model_name}.py"
    trainer = GreedyCoordinateGradientAdversarialSuffixGenerator(config_path)
    data_path = "https://raw.githubusercontent.com/llm-attacks/llm-attacks/main/data/advbench/harmful_behaviors.csv"
    result_prefix = f"results/individual_behaviors_{model_name}_gcg_offset_0"
    
    trainer.set_training_params(
        train_data=data_path,
        result_prefix=result_prefix,
        n_train_data=n_train_data,
        n_steps=n_steps,
        test_steps=test_steps
    )
    trainer.generate_suffix()

if __name__ == '__main__':
    run_trainer("mistral", 1, 100)

