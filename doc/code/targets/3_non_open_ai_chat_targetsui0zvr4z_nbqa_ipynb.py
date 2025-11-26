# %%NBQA-CELL-SEP52c935
from pyrit.executor.attack import ConsoleAttackResultPrinter, PromptSendingAttack
from pyrit.prompt_target import AzureMLChatTarget
from pyrit.setup import IN_MEMORY, initialize_pyrit

initialize_pyrit(memory_db_type=IN_MEMORY)

# Defaults to endpoint and api_key pulled from the AZURE_ML_MANAGED_ENDPOINT and AZURE_ML_KEY environment variables
azure_ml_chat_target = AzureMLChatTarget()

# The environment variable args can be adjusted below as needed for your specific model.
azure_ml_chat_target._set_env_configuration_vars(
    endpoint_uri_environment_variable="AZURE_ML_MANAGED_ENDPOINT", api_key_environment_variable="AZURE_ML_KEY"
)
# Parameters such as temperature and repetition_penalty can be set using the _set_model_parameters() function.
azure_ml_chat_target._set_model_parameters(temperature=0.9, repetition_penalty=1.3)

attack = PromptSendingAttack(objective_target=azure_ml_chat_target)

result = await attack.execute_async(objective="Hello! Describe yourself and the company who developed you.")  # type: ignore
await ConsoleAttackResultPrinter().print_conversation_async(result=result)  # type: ignore

azure_ml_chat_target.dispose_db_engine()
