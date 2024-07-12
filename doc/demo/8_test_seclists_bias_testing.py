# %%
# Import necessary packages
import os
import time
import uuid
import logging
from pyrit.orchestrator.prompt_sending_orchestrator import PromptSendingOrchestrator
from pyrit.common import default_values
from pyrit.prompt_target import OpenAIChatTarget
from pyrit.datasets.fetch_examples import fetch_seclists_bias_testing_examples
from pyrit.score.self_ask_likert_scorer import SelfAskLikertScorer, LikertScalePaths

# %%
# Set logging level to ERROR to suppress informational and warning logs
logging.basicConfig(level=logging.ERROR)

# Set logging level for not seeing logs from specific libraries
logging.getLogger('pyrit').setLevel(logging.ERROR)
logging.getLogger('httpx').setLevel(logging.ERROR)

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

    # Set up the Bias Scorer with the same prompt target to ensure consistency
    bias_scorer = SelfAskLikertScorer(likert_scale_path=LikertScalePaths.FAIRNESS_BIAS_SCALE.value, chat_target=prompt_target)

    # Create the orchestrator with scorer
    orchestrator = PromptSendingOrchestrator(prompt_target=prompt_target, scorers=[bias_scorer])

    # Fetch examples from SecLists Bias Testing dataset
    prompt_dataset  = fetch_seclists_bias_testing_examples(examples_source, source_type='repository')

    # Number of examples we want to use
    prompt_list = prompt_dataset.prompts[:5]

    # Store original prompts and responses
    response_data = {}

    # Assign unique IDs to each prompt
    prompt_requests = []
    for prompt in prompt_list:
        prompt_id = str(uuid.uuid4()) 
        prompt_requests.append({
            "id": prompt_id,
            "prompt": prompt
        })
  
    # Implement simple retry logic
    max_retries = 5
    for attempt in range(max_retries):
        try:
            responses = await orchestrator.send_prompts_async(prompt_list=[req["prompt"] for req in prompt_requests])
            break  # Exit loop if request is successful
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
            else:
                raise

    # Capture the responses with their IDs
    for req, response in zip(prompt_requests, responses):
        for piece in response.request_pieces:
            response_data[str(piece.id)] = {
                "original_prompt": req["prompt"],
                "response": piece.converted_value  # Assuming converted_value is where the LLM response is stored
            }

    # Retrieve the memory to print scoring results
    memory = orchestrator.get_memory()
    score_memory = orchestrator.get_score_memory()

    for entry in memory:
        response_info = response_data.get(str(entry.id), {"original_prompt": "Unknown prompt", "response": "Unknown response"})
        for score_entry in score_memory:
            if entry.id == score_entry.prompt_request_response_id:
                print(
                    f"Original prompt: {response_info['original_prompt']}\n"
                    f"LLM response: {response_info['response']}\n"
                    f"Score category: {score_entry.score_category}\n"
                    f"Score value: {score_entry.get_value()}\n"
                    f"Score rationale: {score_entry.score_rationale}\n\n"
                )

# %%
# Run the main function
if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
