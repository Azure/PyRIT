# %%NBQA-CELL-SEP52c935
# Import the Azure ML SDK components required for workspace connection and model management.
import os
from typing import Union

# Import necessary libraries for Azure ML operations and authentication
from azure.ai.ml import MLClient, UserIdentityConfiguration
from azure.ai.ml.dsl import pipeline
from azure.ai.ml.entities import AmlCompute
from azure.core.exceptions import ResourceNotFoundError
from azure.identity import DefaultAzureCredential, InteractiveBrowserCredential
from dotenv import load_dotenv

# %%NBQA-CELL-SEP52c935
# Load the environment variables from the .env file
load_dotenv()

subscription_id = os.getenv("AZURE_SUBSCRIPTION_ID")
resource_group = os.getenv("AZURE_RESOURCE_GROUP")
workspace_name = os.getenv("AZURE_ML_WORKSPACE_NAME")
registry_name = os.getenv("AZURE_ML_REGISTRY_NAME")
aml_import_model_version = os.getenv("AZURE_ML_MODEL_IMPORT_VERSION")  # values could be 'latest' or any version

# Model and Compute Configuration
model_id = os.getenv("HF_MODEL_ID")
task_name = os.getenv("TASK_NAME")
aml_compute_type = os.getenv("AZURE_ML_COMPUTE_TYPE")
instance_size = os.getenv("AZURE_ML_INSTANCE_SIZE")
compute_name = os.getenv("AZURE_ML_COMPUTE_NAME")
experiment_name = f"Import Model Pipeline Hugging Face model {model_id}"
min_instances = int(os.getenv("AZURE_ML_MIN_INSTANCES"))
max_instances = int(os.getenv("AZURE_ML_MAX_INSTANCES"))
idle_time_before_scale_down = int(os.getenv("IDLE_TIME_BEFORE_SCALE_DOWN"))

# %%NBQA-CELL-SEP52c935
print(f"Subscription ID: {subscription_id}")
print(f"Resource group: {resource_group}")
print(f"Workspace name: {workspace_name}")
print(f"Registry name: {registry_name}")
print(f"Azure ML import model version: {aml_import_model_version}")
print(f"Model ID: {model_id}")
print(f"Task name: {task_name}")
print(f"Azure ML compute type: {aml_compute_type}")
print(f"Instance size: {instance_size}")
print(f"Compute name: {compute_name}")
print(f"Experiment name: {experiment_name}")
print(f"Min instance count: {min_instances}")
print(f"Max instance count: {max_instances}")
print(f"Idle time before scale down in seconds: {idle_time_before_scale_down}")

# %%NBQA-CELL-SEP52c935
# Setup Azure credentials, preferring DefaultAzureCredential and falling back to InteractiveBrowserCredential if necessary
try:
    credential: Union[DefaultAzureCredential, InteractiveBrowserCredential] = DefaultAzureCredential()
    # Verify if the default credential can fetch a token successfully
    credential.get_token("https://management.azure.com/.default")
except Exception as ex:
    print("DefaultAzureCredential failed, falling back to InteractiveBrowserCredential:")
    credential = InteractiveBrowserCredential()

# %%NBQA-CELL-SEP52c935
# Initialize MLClient for Azure ML workspace and registry access
try:
    # Attempt to create MLClient using configuration file
    ml_client_ws = MLClient.from_config(credential=credential)
except:
    ml_client_ws = MLClient(
        credential,
        subscription_id=subscription_id,
        resource_group_name=resource_group,
        workspace_name=workspace_name,
    )
# Initialize MLClient for Azure ML model registry access
ml_client_registry = MLClient(credential, registry_name=registry_name)

# %%NBQA-CELL-SEP52c935
# Setup or retrieve the compute target for model training
try:
    # Check if the compute target already exists
    _ = ml_client_ws.compute.get(compute_name)
    print("Found existing compute target.")
except ResourceNotFoundError:
    # If not found, create a new compute target
    print("Creating a new compute target...")
    compute_config = AmlCompute(
        name=compute_name,
        type=aml_compute_type,
        size=instance_size,
        min_instances=min_instances,
        max_instances=max_instances,
        idle_time_before_scale_down=idle_time_before_scale_down,
    )
    ml_client_ws.begin_create_or_update(compute_config).result()

# %%NBQA-CELL-SEP52c935
import_model = ml_client_registry.components.get(name="import_model", version=aml_import_model_version)

# %%NBQA-CELL-SEP52c935
def get_max_model_version(models: list) -> str:
    """
    Finds the maximum model version number in the given list of models.

    Args:
        models (list): A list of model objects, each having a 'version' attribute as a string.

    Returns:
        str: The maximum version number found among the models as a string.
    """

    # Find the model with the maximum version number
    max_version = max(models, key=lambda x: int(x.version)).version
    model_max_version = str(int(max_version))
    return model_max_version

# %%NBQA-CELL-SEP52c935
def check_model_in_registry(client, model_id: str):
    """
    Checks for the existence of a model with the specified model_id in the given client registry and retrieves its maximum version.

    This function lists all models with the given name in the registry using the provided client. If one or more models are found,
    it determines the model with the highest version number and returns that version. If no models are found, it indicates that the model
    does not exist in the registry.

    Args:
        client: The client object used to interact with the model registry. This can be an Azure ML model catalog client or an Azure ML workspace model client.
        model_id (str): The unique identifier of the model to check in the registry.

    Returns:
        tuple:
            - bool: True if the model exists in the registry, False otherwise.
            - str: The maximum version of the model found in the registry as a string. Returns '0' if the model is not found.
    """
    models = list(client.models.list(name=model_id))
    if models:
        model_version = get_max_model_version(models)
        return True, model_version
    return False, "0"

# %%NBQA-CELL-SEP52c935
# Check if Hugging Face model exists in both the Azure ML Hugging Face model catalog registry and the AML workspace model registry
# Initially assume the model does not exist in either registry
huggingface_model_exists_in_aml_registry = False
registered_model_id = model_id.replace("/", "-")  # model name in registry doesn't contain '/'
try:
    # Check in Azure ML model catalog registry
    exists_in_catalog, catalog_version = check_model_in_registry(ml_client_registry, registered_model_id)
    if exists_in_catalog:
        print(
            f"Model already exists in Azure ML model catalog with name {registered_model_id} and maximum version {catalog_version}"
        )
        huggingface_model_exists_in_aml_registry = True
    else:
        # If not found in the model catalog, check in AML workspace model registry
        exists_in_workspace, workspace_version = check_model_in_registry(ml_client_ws, registered_model_id)
        if exists_in_workspace:
            print(
                f"Model already exists in Azure ML workspace model registry with name {registered_model_id} and maximum version {workspace_version}"
            )
            huggingface_model_exists_in_aml_registry = True

    # If the model doesn't exist in either registry, indicate it needs to be imported
    if not huggingface_model_exists_in_aml_registry:
        print(f"Model {registered_model_id} not found in any registry. Proceeding with model import.")

except Exception as e:
    print(f"Model {registered_model_id} not found in registry. Please continue importing the model.")

# %%NBQA-CELL-SEP52c935
# Define a Azure ML pipeline for importing models into Azure ML


@pipeline
def model_import_pipeline(model_id, compute, task_name, instance_type):
    """
    Pipeline to import a model into Azure ML.

    Parameters:
    - model_id: The ID of the model to import.
    - compute: The compute resource to use for the import job.
    - task_name: The task associated with the model.
    - instance_type: The type of instance to use for the job.

    Returns:
    - A dictionary containing model registration details.
    """
    import_model_job = import_model(
        model_id=model_id, compute=compute, task_name=task_name, instance_type=instance_type
    )
    import_model_job.settings.continue_on_step_failure = False  # Do not continue on failure

    return {"model_registration_details": import_model_job.outputs.model_registration_details}

# %%NBQA-CELL-SEP52c935
# Configure the pipeline object with necessary parameters and identity
pipeline_object = model_import_pipeline(
    model_id=model_id, compute=compute_name, task_name=task_name, instance_type=instance_size
)
pipeline_object.identity = UserIdentityConfiguration()
pipeline_object.settings.force_rerun = True
pipeline_object.settings.default_compute = compute_name

# %%NBQA-CELL-SEP52c935
# Determine if the pipeline needs to be scheduled for model import
schedule_huggingface_model_import = (
    not huggingface_model_exists_in_aml_registry and model_id not in [None, "None"] and len(model_id) > 1
)
print(f"Need to schedule run for importing {model_id}: {schedule_huggingface_model_import}")

# %%NBQA-CELL-SEP52c935
# Submit and monitor the pipeline job if model import is scheduled
if schedule_huggingface_model_import:
    huggingface_pipeline_job = ml_client_ws.jobs.create_or_update(pipeline_object, experiment_name=experiment_name)
    ml_client_ws.jobs.stream(huggingface_pipeline_job.name)  # Stream logs to monitor the job
