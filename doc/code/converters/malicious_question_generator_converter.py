# %% [markdown]
# # Generating Malicious Questions Using a Custom Converter
#
# In this script, we demonstrate how to use the `MaliciousQuestionGeneratorConverter` to generate malicious questions based on an initial prompt about cybersecurity vulnerabilities.
# The converter interacts with the Azure OpenAI API, sending prompts asynchronously through the `PromptSendingOrchestrator`.
#
# This script shows how to:
# - Initialize and configure environment variables for the OpenAI API.
# - Use the `MaliciousQuestionGeneratorConverter` to generate multiple malicious questions from a single prompt.
# - Send the generated prompts asynchronously using `PromptSendingOrchestrator`.
# - Print the conversations and responses from the API.


# %%
import os
from pyrit.common import default_values
from pyrit.prompt_target import OpenAIChatTarget
from pyrit.orchestrator import PromptSendingOrchestrator
from pyrit.prompt_converter.malicious_question_generator_converter import MaliciousQuestionGeneratorConverter  # Import your custom converter

# Load default environment values (API keys, endpoints, etc.)
default_values.load_default_env()

# Define the prompts
prompts = ["cybersecurity vulnerabilities"]

# Initialize Azure OpenAI completion target
prompt_target = OpenAIChatTarget(
    api_key=os.environ.get("OPENAI_CHAT_KEY"),
    endpoint=os.environ.get("OPENAI_CHAT_ENDPOINT"),
    deployment_name=os.environ.get("UNCENSORED_OPENAI_CHAT_DEPLOYMENT"),
)

# Initialize the MaliciousQuestionGeneratorConverter
malicious_question_converter = MaliciousQuestionGeneratorConverter(target=prompt_target, max_iterations=5)

# Initialize the orchestrator
with PromptSendingOrchestrator(
    prompt_target=prompt_target,  # The target to which the prompt will be sent (e.g., Azure OpenAI or OpenAI)
    prompt_converters=[malicious_question_converter]  # Stack of converters to apply (only using the malicious question converter)
) as orchestrator:
    # Send the prompts asynchronously through the orchestrator
    await orchestrator.send_prompts_async(prompt_list=prompts)  # type: ignore

    orchestrator.print_conversations()  # Print the conversations
