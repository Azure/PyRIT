import os
from train import GreedyCoordinateGradientAdversarialSuffixGenerator

def run_trainer(model_name, setup = "behaviors", n_train_data=1, n_steps=100, test_steps=25, data_offset=0, batch_size=128):
    """
    Trains and generates adversarial suffix - single model single prompt

    Args:
        model_name (str): The name of the model, currently supports "mistral" and "llama2"
        setup (str): Identifier for the setup, currently supporst "behavors"
        n_train_data (int): Number of training data instances to use. Default is 1.
        n_steps (int): Number of training steps to perform. Default is 100.
        test_steps (int): Number of training steps after which a test is performed. Default is 25.
        data_offset (int): Offset index to start from in the training data. Default is 0.
        batch_size (int): Size of the batch used in training. Default is 128.
    """

    config_path = f"configs/individual_{model_name}.py"
    trainer = GreedyCoordinateGradientAdversarialSuffixGenerator(config_path)
    train_data_path = f"https://raw.githubusercontent.com/llm-attacks/llm-attacks/main/data/advbench/harmful_{setup}.csv"
    result_prefix = f"results/individual_{setup}_{model_name}_gcg_offset_{data_offset}"

    if not os.path.exists('results'):
        os.makedirs('results')

    trainer.generate_suffix(
        train_data=train_data_path,
        result_prefix=result_prefix,
        n_train_data=n_train_data,
        n_steps=n_steps,
        test_steps=test_steps,
        data_offset = data_offset,
        batch_size = batch_size)

if __name__ == '__main__':
    run_trainer(model_name = "mistral", setup = "behaviors")

