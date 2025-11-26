# %%NBQA-CELL-SEP52c935
import time

from pyrit.executor.attack import (
    AttackExecutor,
    ConsoleAttackResultPrinter,
    PromptSendingAttack,
)
from pyrit.prompt_target import HuggingFaceChatTarget
from pyrit.setup import IN_MEMORY, initialize_pyrit

initialize_pyrit(memory_db_type=IN_MEMORY)

# models to test
model_id = "Qwen/Qwen2-0.5B-Instruct"

# List of prompts to send
prompt_list = ["What is 3*3? Give me the solution.", "What is 4*4? Give me the solution."]

# Dictionary to store average response times
model_times = {}

print(f"Running model: {model_id}")

try:
    # Initialize HuggingFaceChatTarget with the current model
    target = HuggingFaceChatTarget(model_id=model_id, use_cuda=False, tensor_format="pt", max_new_tokens=30)

    # Initialize the attack
    attack = PromptSendingAttack(objective_target=target)

    # Record start time
    start_time = time.time()

    # Send prompts asynchronously
    responses = await AttackExecutor().execute_multi_objective_attack_async(  # type: ignore
        attack=attack,
        objectives=prompt_list,
    )

    # Record end time
    end_time = time.time()

    # Calculate total and average response time
    total_time = end_time - start_time
    avg_time = total_time / len(prompt_list)
    model_times[model_id] = avg_time

    print(f"Average response time for {model_id}: {avg_time:.2f} seconds\n")

    # Print the conversations
    for result in responses:
        await ConsoleAttackResultPrinter().print_conversation_async(result=result)  # type: ignore

except Exception as e:
    print(f"An error occurred with model {model_id}: {e}\n")
    model_times[model_id] = None

# Print the model average time
if model_times[model_id] is not None:
    print(f"{model_id}: {model_times[model_id]:.2f} seconds")
else:
    print(f"{model_id}: Error occurred, no average time calculated.")

# %%NBQA-CELL-SEP52c935
from pyrit.memory import CentralMemory

memory = CentralMemory.get_memory_instance()
memory.dispose_engine()
