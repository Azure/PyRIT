# Populating Secrets - Quick Start Guide

Before running PyRIT, you need to configure access to AI targets. This guide will help you get started quickly.

## Fastest Way to Get Started

The simplest way to configure PyRIT requires just two environment variables and three lines of code:

```python
from pyrit.setup import initialize_pyrit
from pyrit.setup.initializers import SimpleInitializer

initialize_pyrit(memory_db_type="InMemory", initializers=[SimpleInitializer()])
```

This sets up PyRIT with sensible defaults using in-memory storage. You just need to set two environment variables:
- `OPENAI_CHAT_ENDPOINT` - Your AI endpoint URL
- `OPENAI_CHAT_KEY` - Your API key

With this setup, you can run most PyRIT notebooks and examples!

## Setting Up Environment Variables

PyRIT loads secrets and endpoints from environment variables or a `.env` file in your repo root. The `.env_example` file shows the format and available options.

### Creating Your .env File

1. Copy `.env_example` to `.env` in your repository root
2. Add your API credentials. For example, for Azure OpenAI:

```bash
OPENAI_CHAT_ENDPOINT="https://your-resource.openai.azure.com/openai/deployments/your-deployment/chat/completions"
OPENAI_CHAT_KEY="your-api-key-here"
```

To find these values in Azure Portal: `Azure Portal > Azure AI Services > Azure OpenAI > Your OpenAI Resource > Resource Management > Keys and Endpoint`

### Using .env.local for Overrides

You can use `.env.local` to override values in `.env` without modifying the base file. This is useful for:
- Testing different targets
- Using personal credentials instead of shared ones
- Switching between configurations quickly

Simply create `.env.local` and add any variables you want to override. PyRIT will prioritize `.env.local` over `.env`.

## Authentication Options

### API Keys (Default)
The simplest approach is using API keys as shown above. Most targets support this method.

### Azure Entra Authentication (Optional)
For Azure resources, you can use Entra auth instead of API keys. This requires:

1. Install Azure CLI for your OS:
   - [Windows](https://learn.microsoft.com/en-us/cli/azure/install-azure-cli-windows?tabs=azure-cli)
   - [Linux](https://learn.microsoft.com/en-us/cli/azure/install-azure-cli-linux?pivots=apt)
   - [macOS](https://learn.microsoft.com/en-us/cli/azure/install-azure-cli-macos)

2. Log in to Azure:
   ```bash
   az login
   ```

When using Entra auth, you don't need to set API keys for Azure resources.

## Next Steps

- For detailed configuration options, see the [Configuration Guide](../code/setup/0_configuration.ipynb)
- For database options beyond in-memory storage, see the [Memory Documentation](../code/memory/0_memory.md)
