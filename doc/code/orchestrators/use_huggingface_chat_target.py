# %% [markdown]
# # HuggingFace Chat Target Testing
#
# This notebook demonstrates the process of testing the HuggingFace Chat Target using various prompts.
# The target model will be loaded and interacted with using different prompts to examine its responses.
#
# ### Key Points:
#
# 1. **Supported Instruction Models**:
#    - This notebook supports the following **instruct models** that follow a structured chat template.
#      These are examples, and more instruct models are available on Hugging Face:
#      - `HuggingFaceTB/SmolLM-360M-Instruct`
#      - `microsoft/Phi-3-mini-4k-instruct`
#      - `...`
#
# 2. **Excluded Models**:
#    - Non-instruct models (e.g., `"google/gemma-2b"`, `"princeton-nlp/Sheared-LLaMA-1.3B-ShareGPT"`)
#      are **not included** in this demo, as they do not follow the structured chat template required for the current local Hugging Face model support.
#
# 3. **Model Response Times**:
#    - The tests were conducted using a CPU, and the following are the average response times for each model:
#      - `HuggingFaceTB/SmolLM-1.7B-Instruct`: 5.87 seconds
#      - `HuggingFaceTB/SmolLM-135M-Instruct`: 3.09 seconds
#      - `HuggingFaceTB/SmolLM-360M-Instruct`: 3.31 seconds
#      - `microsoft/Phi-3-mini-4k-instruct`: 4.89 seconds
#      - `Qwen/Qwen2-0.5B-Instruct`: 1.38 seconds
#      - `Qwen/Qwen2-1.5B-Instruct`: 2.96 seconds
#      - `stabilityai/stablelm-2-zephyr-1_6b`: 5.31 seconds
#      - `stabilityai/stablelm-zephyr-3b`: 8.37 seconds
#
# 4. **Model Output Behavior**:
#      - During testing, it's been observed that some models may continue generating responses even after answering the prompt.
#        This can result in incomplete or truncated responses due to the max_new_token limit setting, which restricts the number of tokens generated.
#        When interacting with these models, it's important to be aware that responses might get cut off if the output exceeds the token limit.


# %%
import time
from pyrit.prompt_target import HuggingFaceChatTarget
from pyrit.orchestrator import PromptSendingOrchestrator

# List of models to iterate through
model_list = [
    "microsoft/Phi-3-mini-4k-instruct",
    "HuggingFaceTB/SmolLM-135M-Instruct",
    "HuggingFaceTB/SmolLM-360M-Instruct",
    "HuggingFaceTB/SmolLM-1.7B-Instruct",
    "Qwen/Qwen2-0.5B-Instruct",
    "Qwen/Qwen2-1.5B-Instruct",
    "stabilityai/stablelm-2-zephyr-1_6b",
    "stabilityai/stablelm-zephyr-3b",
]

# List of prompts to send
prompt_list = ["What is 3*3?", "What is 4*4?", "What is 5*5?", "What is 6*6?", "What is 7*7?"]


# Dictionary to store average response times
model_times = {}

for model_id in model_list:
    print(f"Running model: {model_id}")

    try:
        # Initialize HuggingFaceChatTarget with the current model
        target = HuggingFaceChatTarget(
            model_id=model_id,  # Use the desired model ID
            use_cuda=False,  # Set to True if using CUDA
            tensor_format="pt",
            max_new_tokens=30,  # Increases response length and time.
        )

        # Initialize the orchestrator
        orchestrator = PromptSendingOrchestrator(prompt_target=target, verbose=False)

        # Record start time
        start_time = time.time()

        # Send prompts asynchronously
        responses = await orchestrator.send_prompts_async(prompt_list=prompt_list)  # type: ignore

        # Record end time
        end_time = time.time()

        # Calculate total and average response time
        total_time = end_time - start_time
        avg_time = total_time / len(prompt_list)
        model_times[model_id] = avg_time

        print(f"Average response time for {model_id}: {avg_time:.2f} seconds\n")

        # Print the conversations
        orchestrator.print_conversations()
        print("-" * 50)

    except Exception as e:
        print(f"An error occurred with model {model_id}: {e}\n")
        model_times[model_id] = None
        continue

# Print all model average times
print("Model Average Response Times:")
for model_id, avg_time in model_times.items():
    if avg_time is not None:
        print(f"{model_id}: {avg_time:.2f} seconds")
    else:
        print(f"{model_id}: Error occurred, no average time calculated.")
