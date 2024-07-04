# %%
# Import necessary packages
import os
import time
from pyrit.orchestrator.prompt_sending_orchestrator import PromptSendingOrchestrator
from pyrit.common import default_values
from pyrit.prompt_target import OpenAIChatTarget
from pyrit.datasets.fetch_examples import fetch_seclists_bias_testing_examples


# %%
# Load environment variables
default_values.load_default_env()

# %%
# Retrieve environment variables
openai_api_key = os.getenv("OPENAI_API_KEY")
deployment_name = os.getenv("OPENAI_CHAT_DEPLOYMENT")
endpoint = os.getenv("OPENAI_ENDPOINT")
examples_source = os.getenv("EXAMPLES_SOURCE")

# %%
async def main():
    # Create orchestrator
    prompt_target = OpenAIChatTarget(api_key=openai_api_key, deployment_name=deployment_name, endpoint=endpoint)
    orchestrator = PromptSendingOrchestrator(prompt_target=prompt_target)

    # Fetch examples from SecLists Bias Testing dataset
    prompt_dataset  = fetch_seclists_bias_testing_examples(examples_source, source_type='repository')
    print("PromptDataset created successfully.")
    # print(prompt_dataset.prompts) # Uncomment to see the examples

    # Number of examples we have
    print(f"Number of examples: {len(prompt_dataset.prompts)}")

    # Number of examples we want to use
    prompt_list = prompt_dataset.prompts[:5]
    print(f"Number of examples to use: {len(prompt_list)}")

    # Send prompt with examples to target
    print("Sending prompt to target...")
    
    # Implement simple retry logic
    max_retries = 5
    for attempt in range(max_retries):
        try:
            response = await orchestrator.send_prompts_async(prompt_list=prompt_list)
            break  # Exit loop if request is successful
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
            else:
                raise

    # Proof of concept to verify response content, can be removed
    orchestrator.print_conversation(response)

# %%
# Run the main function
if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
