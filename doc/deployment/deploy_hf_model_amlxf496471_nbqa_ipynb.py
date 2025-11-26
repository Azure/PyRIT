# %%NBQA-CELL-SEP52c935
import os
import random
import string

from dotenv import load_dotenv

# Load the environment variables from the .env file
load_dotenv()

subscription_id = os.getenv("AZURE_SUBSCRIPTION_ID")
resource_group = os.getenv("AZURE_RESOURCE_GROUP")
workspace_name = os.getenv("AZURE_ML_WORKSPACE_NAME")
registry_name = os.getenv("AZURE_ML_REGISTRY_NAME")
model_to_deploy = os.getenv("AZURE_ML_MODEL_NAME_TO_DEPLOY")
model_version = os.getenv("AZURE_ML_MODEL_VERSION_TO_DEPLOY")
instance_type = os.getenv("AZURE_ML_MODEL_DEPLOY_INSTANCE_SIZE")
instance_count = int(os.getenv("AZURE_ML_MODEL_DEPLOY_INSTANCE_COUNT"))
request_timeout_ms = int(os.getenv("AZURE_ML_MODEL_DEPLOY_REQUEST_TIMEOUT_MS"))
liveness_probe_initial_delay = int(os.getenv("AZURE_ML_MODEL_DEPLOY_LIVENESS_PROBE_INIT_DELAY_SECS"))

# %%NBQA-CELL-SEP52c935
print(f"Subscription ID: {subscription_id}")
print(f"Resource group: {resource_group}")
print(f"Workspace name: {workspace_name}")
print(f"Registry name: {registry_name}")
print(f"Model to deploy: {model_to_deploy}")
print(f"Model version: {model_version}")
print(f"Instance type: {instance_type}")
print(f"Instance count: {instance_count}")
print(f"Request timeout in millis: {request_timeout_ms}")
print(f"Liveness probe initial delay in secs: {liveness_probe_initial_delay}")

# %%NBQA-CELL-SEP52c935
from typing import Union

from azure.ai.ml import MLClient
from azure.core.exceptions import ResourceNotFoundError
from azure.identity import DefaultAzureCredential, InteractiveBrowserCredential

try:
    credential: Union[DefaultAzureCredential, InteractiveBrowserCredential] = DefaultAzureCredential()
    credential.get_token("https://management.azure.com/.default")
except Exception as ex:
    credential = InteractiveBrowserCredential()

workspace_ml_client = MLClient(
    credential, subscription_id=subscription_id, resource_group_name=resource_group, workspace_name=workspace_name
)
registry_ml_client = MLClient(credential, registry_name=registry_name)

# %%NBQA-CELL-SEP52c935


def check_model_version_exists(client, model_name, version) -> bool:
    """
    Checks for the existence of a specific version of a model with the given name in the client registry.

    This function lists all models with the given name in the registry using the provided client. It then checks if the specified version exists among those models.

    Args:
        client: The client object used to interact with the model registry. This can be an Azure ML model catalog client or an Azure ML workspace model client.
        model_name (str): The name of the model to check in the registry.
        version (str): The specific version of the model to check for.

    Returns:
        bool: True if the model with the specified version exists in the registry, False otherwise.
    """
    model_found = False
    try:
        models = list(client.models.list(name=model_name))
        model_found = any(model.version == version for model in models)
    except ResourceNotFoundError as rnfe:
        print(f"Model not found in the registry{registry_name}, please try other registry.")
    return model_found


# %%NBQA-CELL-SEP52c935
# Check if the Hugging Face model exists in the Azure ML workspace model registry
model = None
if check_model_version_exists(workspace_ml_client, model_to_deploy, model_version):
    print("Model found in the Azure ML workspace model registry.")
    model = workspace_ml_client.models.get(model_to_deploy, model_version)
    print(
        "\n\nUsing model name: {0}, version: {1}, id: {2} for inferencing".format(model.name, model.version, model.id)
    )
# Check if the Hugging Face model exists in the Azure ML model catalog registry
elif check_model_version_exists(registry_ml_client, model_to_deploy, model_version):
    print("Model found in the Azure ML model catalog registry.")
    model = registry_ml_client.models.get(model_to_deploy, model_version)
    print(
        "\n\nUsing model name: {0}, version: {1}, id: {2} for inferencing".format(model.name, model.version, model.id)
    )
else:
    raise ValueError(
        f"Model {model_to_deploy} not found in any registry. Please run the notebook (download_and_register_hf_model_aml.ipynb) to download and register Hugging Face model to Azure ML workspace model registry."
    )
endpoint_name = model_to_deploy + str(model_version)

# %%NBQA-CELL-SEP52c935


def get_updated_endpoint_name(endpoint_name):
    """
    Generates a unique string based on the Azure ML endpoint name.

    This function takes the first 26 characters of the given endpoint name and appends
    a 5-character random alphanumeric string with hyphen to ensure uniqueness.
    """
    # Take the first 26 characters of the endpoint name
    base_name = endpoint_name[:26]

    # Generate a 5-char random alphanumeric string and append to '-'
    random_suffix = "-" + "".join(random.choices(string.ascii_letters + string.digits, k=5))

    updated_name = f"{base_name}{random_suffix}"

    return updated_name


# %%NBQA-CELL-SEP52c935
endpoint_name = get_updated_endpoint_name(endpoint_name)

# %%NBQA-CELL-SEP52c935
print(f"Endpoint name: {endpoint_name}")

# %%NBQA-CELL-SEP52c935
from azure.ai.ml.entities import (
    ManagedOnlineDeployment,
    ManagedOnlineEndpoint,
    OnlineRequestSettings,
    ProbeSettings,
)

# create an online endpoint
endpoint = ManagedOnlineEndpoint(
    name=endpoint_name, description=f"Online endpoint for {model_to_deploy}", auth_mode="key"
)
workspace_ml_client.begin_create_or_update(endpoint).wait()

# %%NBQA-CELL-SEP52c935
# create a deployment
# Create probe settings
liveness_probe = ProbeSettings(initial_delay=liveness_probe_initial_delay)
deployment = ManagedOnlineDeployment(
    name=f"{endpoint_name}",
    endpoint_name=endpoint_name,
    model=model.id,
    instance_type=instance_type,
    instance_count=instance_count,
    request_settings=OnlineRequestSettings(request_timeout_ms=request_timeout_ms),
    liveness_probe=liveness_probe,
)
workspace_ml_client.online_deployments.begin_create_or_update(deployment).wait()
workspace_ml_client.begin_create_or_update(endpoint