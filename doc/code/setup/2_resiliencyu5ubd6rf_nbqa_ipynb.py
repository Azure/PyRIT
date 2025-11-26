# %%NBQA-CELL-SEP52c935
from pyrit.prompt_target import OpenAIChatTarget
from pyrit.scenario import FoundryScenario, FoundryStrategy
from pyrit.setup import IN_MEMORY, initialize_pyrit

initialize_pyrit(memory_db_type=IN_MEMORY)

objective_target = OpenAIChatTarget(model_name="gpt-4o")

# Create a scenario with retry configuration
scenario = FoundryScenario()

await scenario.initialize_async( # type: ignore
    objective_target=objective_target,
    max_concurrency=5,
    max_retries=3,
    scenario_strategies=[FoundryStrategy.Base64],
)

# Execute with automatic retry after exceptions
result = await scenario.run_async()  # type: ignore

print(f"Scenario completed after {result.number_tries} attempt(s)")
print(f"Total results: {len(result.attack_results)}")
