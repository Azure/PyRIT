<code>train.py</code> contains the class for generating suffix.

<code>run.py</code> contains a demo for generating a suffix. The results will be saved in the <code>experiments/results</code> directory as JSON logs that contain information such as target (prompt), suffix, and loss.

## Model Supports

Currently we support 5 models:
- microsoft/Phi-3-mini-4k-instruct: https://huggingface.co/microsoft/Phi-3-mini-4k-instruct

- lmsys/vicuna-13b-v1.5: https://huggingface.co/lmsys/vicuna-13b-v1.5

- mistralai/Mistral-7B-Instruct-v0.1: https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1

- meta-llama/Llama-2-7b-chat-hf: https://huggingface.co/meta-llama/Llama-2-7b-chat-hf

- meta-llama/Meta-Llama-3-8B-Instruct: https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct



## Suffix Optimization Setups
There are 2 setups of suffix optimization:

 - "single": Optimizes suffix for one prompt using one model.

 - "multiple": Optimize suffix across multiple prompts using one or more models. When using multiple models, specify the <code>num_train_models</code> in the <code>run_trainer</code> function.

## Function Calls in <code>run.py</code>

The <code>run.py</code> script includes three different function call examples:

- Optimizing 1 prompt with 1 model (vicuna).
- Optimizing 2 prompts with 1 model (mistral).
- Optimizing 25 prompts with 4 models (vicuna, mistral, llama2 and llama3).

## Reference
"[Universal and Transferable Adversarial Attacks on Aligned Language Models](https://arxiv.org/abs/2307.15043)" by Andy Zou, Zifan Wang, Nicholas Carlini, Milad Nasr, J. Zico Kolter, and Matt Fredrikson.

The paper's official Github: https://github.com/llm-attacks/llm-attacks
