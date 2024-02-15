# %% [markdown]
# ## Deploying Hugging Face Models into Azure ML Managed Online Endpoint
# 
# This notebook demonstrates the process of deploying registered models in AML workspace to an AML managed online endpoint for real-time inference.
# 
# [Learn more about AML Managed Online Endpoints](https://learn.microsoft.com/en-us/azure/machine-learning/concept-endpoints-online?view=azureml-api-2)
# 
# ### Prerequisites
# - An Azure account with an active subscription. [Create one for free](https://azure.microsoft.com/free/).
# - An Azure ML workspace set up. [Learn how to set up a workspace](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-manage-workspace).
# - Install the Azure ML client library for Python with pip.
#   ```bash
#      pip install azure-ai-ml
#      pip install azure-identity
#   ```
# - Execute the `az login` command to sign in to your Azure subscription. For detailed instructions, refer to the "Authenticate with Azure Subscription" section in the notebook provided [here](../setup/azure_openai_setup.ipynb)
# - A Hugging Face model should be present in the AML model catalog. If it is missing, execute the [notebook](./download_and_register_hf_model_aml.ipynb) to download and register the Hugging Face model in the AML registry.

# %% [markdown]
# ### Load Environment Variables
# 
# Load necessary environment variables from an `.env` file.
# 
# ### Environment Variables
# 
# For ex., to download the Hugging Face model `cognitivecomputations/Wizard-Vicuna-13B-Uncensored` into your Azure environment, below are the environment variables that needs to be set in `.env` file:
# 
# 1. **AZURE_SUBSCRIPTION_ID**
#    - Obtain your Azure Subscription ID, essential for accessing Azure services.
# 
# 2. **AZURE_RESOURCE_GROUP**
#    - Identify the Resource Group where your Azure Machine Learning (AML) workspace is located.
# 
# 3. **AML_WORKSPACE_NAME**
#    - Specify the name of your AML workspace where the model will be registered.
# 
# 4. **AML_REGISTRY_NAME**
#    - Choose a name for registering the model in your AML workspace, such as "HuggingFace". This helps in identifying if the model already exists in your AML Hugging Face registry.
# 
# 5. **AML_MODEL_NAME_TO_DEPLOY**
#    - If the model is listed in the AML Hugging Face model catalog, then supply the model name as shown in the following image. 
#    <br> <img src="./../../assets/aml_hf_model.png" alt="aml_hf_model.png" height="400"/> <br>
#    - If you intend to deploy the model from the AML workspace model registry, then use the model name as shown in the subsequent image.
#    <br> <img src="./../../assets/aml_ws_model.png" alt="aml_ws_model.png" height="400"/> <br>
# 6. **AML_MODEL_VERSION_TO_DEPLOY**
#    - You can find the details of the model version in the images from previous step associated with the respective model.
# 
# 7. **AML_MODEL_DEPLOY_INSTANCE_SIZE**
#    - Select the size of the compute instance of for deploying the model, ensuring it's at least double the size of the model to effective inference.
# 
# 9. **AML_MODEL_DEPLOY_INSTANCE_COUNT**
#    - Number of compute instances for model deployment.
# 
# 10. **AML_MODEL_DEPLOY_REQUEST_TIMEOUT_MS**
#     - Set the AML inference endpoint request timeout, recommended value is 60000 (in millis).
# 
# 

# %%

from dotenv import load_dotenv
import os

# Load the environment variables from the .env file
load_dotenv()

subscription_id = os.getenv('AZURE_SUBSCRIPTION_ID')
resource_group = os.getenv('AZURE_RESOURCE_GROUP')
workspace_name = os.getenv('AML_WORKSPACE_NAME')
registry_name = os.getenv('AML_REGISTRY_NAME')
model_to_deploy = os.getenv('AML_MODEL_NAME_TO_DEPLOY')
model_version = os.getenv("AML_MODEL_VERSION_TO_DEPLOY")
instance_type = os.getenv('AML_MODEL_DEPLOY_INSTANCE_SIZE')
instance_count = os.getenv('AML_MODEL_DEPLOY_INSTANCE_COUNT')
request_timeout_ms = os.getenv('AML_MODEL_DEPLOY_REQUEST_TIMEOUT_MS')

# %%
print(f"Subscription ID: {subscription_id}")
print(f"Resource group: {resource_group}")
print(f"Workspace name: {workspace_name}")
print(f"Registry name: {registry_name}")
print(f"Model to deploy: {model_to_deploy}")
print(f"Instance type: {instance_type}")
print(f"Instance count: {instance_count}")
print(f"Request timeout in millis: {request_timeout_ms}")

# %% [markdown]
# ### Configure Credentials
# 
# Set up the `DefaultAzureCredential` for seamless authentication with Azure services. This method should handle most authentication scenarios. If you encounter issues, refer to the [Azure Identity documentation](https://docs.microsoft.com/en-us/python/api/azure-identity/azure.identity?view=azure-python) for alternative credentials.
# 

# %%
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential, InteractiveBrowserCredential
from azure.ai.ml.entities import AmlCompute
from azure.core.exceptions import ResourceNotFoundError

try:
    credential = DefaultAzureCredential()
    credential.get_token("https://management.azure.com/.default")
except Exception as ex:
    credential = InteractiveBrowserCredential()

workspace_ml_client = MLClient(
    credential,
    subscription_id=subscription_id,
    resource_group_name=resource_group,
    workspace_name=workspace_name
)
registry_ml_client = MLClient(credential, registry_name=registry_name)

# %%
def check_model_version_exists(client, model_name, version) -> bool:
    """
    Checks for the existence of a specific version of a model with the given name in the client registry.

    This function lists all models with the given name in the registry using the provided client. It then checks if the specified version exists among those models.

    Args:
        client: The client object used to interact with the model registry. This can be an Azure ML model catalog client or an AML workspace model client.
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
        print("Model not found in the registry")
    return model_found


# %%
# Check if the Hugging Face model exists in the AML workspace model registry
model = None
if check_model_version_exists(workspace_ml_client, model_to_deploy, model_version):
    print("Model found in the AML workspace model registry.")
    model = workspace_ml_client.models.get(model_to_deploy, model_version)
    print(
        "\n\nUsing model name: {0}, version: {1}, id: {2} for inferencing".format(
            model.name, model.version, model.id
        )
    )
# Check if the Hugging Face model exists in the AML model catalog registry
elif check_model_version_exists(registry_ml_client, model_to_deploy, model_version):
    print("Model found in the AML model catalog registry.")
    model = registry_ml_client.models.get(model_to_deploy, model_version)
    print(
        "\n\nUsing model name: {0}, version: {1}, id: {2} for inferencing".format(
            model.name, model.version, model.id
        )
    )
else:
    raise ValueError(f"Model {model_to_deploy} not found in any registry. Please run the notebook (download_and_register_hf_model_aml.ipynb) to download and register Hugging Face model to AML workspace model registry.")
endpoint_name = model_to_deploy + str(model_version)

# %%
endpoint_name

# %%
# Using the first 32 characters because AML endpoint names must be between 3 and 32 characters in length.
endpoint_name = endpoint_name[:32]

# %%
print(endpoint_name)

# %% [markdown]
# **Create an AML managed online endpoint**
# To define an endpoint, you need to specify:
# 
# Endpoint name: The name of the endpoint. It must be unique in the Azure region. For more information on the naming rules, see managed online endpoint limits.
# Authentication mode: The authentication method for the endpoint. Choose between key-based authentication and Azure Machine Learning token-based authentication. A key doesn't expire, but a token does expire. 

# %%
from azure.ai.ml.entities import (
    ManagedOnlineEndpoint,
    ManagedOnlineDeployment,
    OnlineRequestSettings,
)

# create an online endpoint
endpoint = ManagedOnlineEndpoint(name=endpoint_name, description=f"Online endpoint for {model_to_deploy}", auth_mode="key")
workspace_ml_client.begin_create_or_update(endpoint).wait()

# %% [markdown]
# **Add deployment to an AML endpoint created above**
# 
# Please be aware that deploying, particularly larger models, may take some time. Once the deployment is finished, the provisioning state will be marked as 'Succeeded', as illustrated in the image below.
# ![image.png](attachment:image.png)
# <br> <img src="./../../assets/aml_endpoint_deployment.png" alt="aml_endpoint_deployment.png" height="400"/> <br>

# %%
# create a deployment
deployment = ManagedOnlineDeployment(
    name=f"{endpoint_name}",
    endpoint_name=endpoint_name,
    model=model.id,
    instance_type=instance_type,
    instance_count=instance_count,
    request_settings=OnlineRequestSettings(
        request_timeout_ms=60000,
    ),
)
workspace_ml_client.online_deployments.begin_create_or_update(deployment).wait()
# online endpoints can have multiple deployments with traffic split or shadow traffic. Set traffic to 100% for deployment
deployment.traffic = {f"{model_to_deploy}": 100}
workspace_ml_client.begin_create_or_update(endpoint).result()

# %%



