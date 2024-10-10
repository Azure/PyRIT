
from pyrit.common import default_values
from pyrit.datasets import fetch_tdc23_redteaming_dataset
from pyrit.orchestrator import PromptSendingOrchestrator
from pyrit.prompt_target import TextTarget

# Load environment variables
default_values.load_default_env()

# Set up the Azure OpenAI prompt target
prompt_target = TextTarget()

# Create the orchestrator with scorer without safe prompts included
orchestrator = PromptSendingOrchestrator(prompt_target=prompt_target)

# Fetch only unsafe prompts from tdc23_redteaming dataset
prompt_dataset = fetch_tdc23_redteaming_dataset()

# Use the first 8 examples for red teaming
prompt_list = prompt_dataset.prompts[:8]

# Send prompts using the orchestrator and capture responses
try:
    responses = await orchestrator.send_prompts_async(prompt_list=prompt_list)  # type: ignore
    if responses:
        orchestrator.print_conversations()  # Retrieve the memory to print scoring results
    else:
        print("No valid responses were received from the orchestrator.")
except Exception as e:
    print(f"An error occurred while sending prompts: {e}")