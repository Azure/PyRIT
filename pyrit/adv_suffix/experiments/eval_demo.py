import os
from PyRIT.pyrit.adv_suffix.experiments.eval import AdversarialSuffixEvaluator

def run_eval(model_name, logdir = "results/", setup="behaviors", batch_size=128):
    """
    Evaluate suffixes generated - single model single prompt

    Args:
        model_name (str): The name of the model, should be consistent with the name used in suffix genearation
        setup (str): Identifier for the setup, should be consistent with the name used in suffix genearation
        batch_size (int): Size of the batch used in training. Default is 128.
    """
   
    config_path = f"configs/transfer_{model_name}.py"
    evaluator = AdversarialSuffixEvaluator(config_path)
    evaluator.parse_logs(logdir, setup)

    if not os.path.exists('eval'):
        os.makedirs('eval')
    train_data_path = f"https://raw.githubusercontent.com/llm-attacks/llm-attacks/main/data/advbench/harmful_{setup}.csv"
    evaluator.evaluate_suffix(model_name, train_data=train_data_path, batch_size=batch_size)


if __name__ == '__main__':
    run_eval(model_name = "mistral")