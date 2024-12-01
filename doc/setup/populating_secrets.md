# Populating Secrets

Nearly all of PyRIT's targets require secrets to interact with.

PyRIT primarily uses these by putting them in a local `.env` file. In typical AI red team operations, operators may create new targets that require additional environment variables, which might differ from those in the base `.env` file. In such cases, you can place these additional or modified variables in a `.env.local` file, which will take precedence over the base `.env`.

When dealing with Azure OpenAI, you need to have an Azure account and a subscription. Populate the `.env` file in your repo with the correct Azure OpenAI Keys, deployment names, and endpoints.

These are detailed in `.env_example`, and are generally retrievable from the Azure Portal. For example, for Azure OpenAI, you can find these in `Azure Portal > Azure AI Services > Azure OpenAI > Your OpenAI Resource > Resource Management > Keys and Endpoint`

## Authenticate with Azure Subscription

There are certain targets that can interact using AAD auth (e.g. most Azure OpenAI targets). To use this, you must authenticate to your Azure subscription. Depending on your operating system, download the appropriate Azure CLI tool from the links provided below:

   - Windows OS: [Download link](https://learn.microsoft.com/en-us/cli/azure/install-azure-cli-windows?tabs=azure-cli)
   - Linux: [Download link](https://learn.microsoft.com/en-us/cli/azure/install-azure-cli-linux?pivots=apt)
   - Mac OS: [Download link](https://learn.microsoft.com/en-us/cli/azure/install-azure-cli-macos)

   After downloading and installing the Azure CLI, open your terminal and run the following command to log in:

   ```bash
   az login
   ```
