# %%NBQA-CELL-SEP52c935
import os

# Enter details of your AML workspace
subscription_id = os.environ.get("AZURE_SUBSCRIPTION_ID")
resource_group = os.environ.get("AZURE_RESOURCE_GROUP")
workspace = os.environ.get("AZURE_ML_WORKSPACE_NAME")
print(workspace)

# %%NBQA-CELL-SEP52c935
from azure.ai.ml import MLClient
from azure.identity import AzureCliCredential

# Get a handle to the workspace
# For some people DefaultAzureCredential may work better than AzureCliCredential.
ml_client = MLClient(AzureCliCredential(), subscription_id, resource_group, workspace)

# %%NBQA-CELL-SEP52c935
from pathlib import Path

from azure.ai.ml.entities import BuildContext, Environment, JobResourceConfiguration

from pyrit.common.path import HOME_PATH

# Configure the AML environment with path to Dockerfile and dependencies
env_docker_context = Environment(
    build=BuildContext(path=Path(HOME_PATH) / "pyrit" / "auxiliary_attacks" / "gcg" / "src"),
    name="pyrit",
    description="PyRIT environment created from a Docker context.",
)

# Create or update the AML environment
ml_client.environments.create_or_update(env_docker_context)

# %%NBQA-CELL-SEP52c935
from azure.ai.ml import command

# Configure the command
job = command(
    code=Path(HOME_PATH),
    command="cd pyrit/auxiliary_attacks/gcg/experiments && python run.py --model_name ${{inputs.model_name}} --setup ${{inputs.setup}} --n_train_data ${{inputs.n_train_data}} --n_test_data ${{inputs.n_test_data}} --n_steps ${{inputs.n_steps}} --batch_size ${{inputs.batch_size}}",
    inputs={
        "model_name": "phi_3_mini",
        "setup": "multiple",
        "n_train_data": 25,
        "n_test_data": 0,
        "n_steps": 500,
        "batch_size": 256,
    },
    environment=f"{env_docker_context.name}:{env_docker_context.version}",
    environment_variables={"HUGGINGFACE_TOKEN": os.environ["HUGGINGFACE_TOKEN"]},
    display_name="suffix_generation",
    description="Generate a suffix for attacking LLMs.",
    resources=JobResourceConfiguration(
        instance_type="Standard_NC96ads_A100_v4",
        instance_count=1,
    )
)

# %%NBQA-CELL-SEP52c935
# Submit the command
returned_job = ml_client.create_or_update(job)
