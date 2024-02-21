### Setup

When dealing with Azure OpenAI, you need to have an Azure account and a subscription. Populate the `.env` file in your repo with the correct Azure Open AI Keys, deployment names, and endpoints.

You can find these in `Azure Portal > Azure AI Services > Azure OpenAI > Your OpenAI Resource > Resource Management > Keys and Endpoint`

### Authenticate with Azure Subscription
To begin interacting with Azure resources deployed in your Azure Subscription, you need to authenticate first. Depending on your operating system, download the appropriate Azure CLI tool from the links provided below:
   - Windows OS: [Download link](https://learn.microsoft.com/en-us/cli/azure/install-azure-cli-windows?tabs=azure-cli)
   - Linux: [Download link](https://learn.microsoft.com/en-us/cli/azure/install-azure-cli-linux?pivots=apt)
   - Mac OS: [Download link](https://learn.microsoft.com/en-us/cli/azure/install-azure-cli-macos)

   After downloading and installing the Azure CLI, open your terminal and run the following command to log in:

   ```bash
   az login
   ```
