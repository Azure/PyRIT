# Populating Secrets - Quick Start Guide

Before running PyRIT, you need to configure access to AI targets. This guide will help you get started quickly.

## Fastest Way to Get Started

The simplest way to configure PyRIT requires just two environment variables and three lines of code:

```python
from pyrit.setup import initialize_pyrit_async
from pyrit.setup.initializers import SimpleInitializer

await initialize_pyrit_async(memory_db_type="InMemory", initializers=[SimpleInitializer()])
```

This sets up PyRIT with sensible defaults using in-memory storage. You just need to set two environment variables:
- `OPENAI_CHAT_ENDPOINT` - Your AI endpoint URL
- `OPENAI_CHAT_KEY` - Your API key

With this setup, you can run most PyRIT notebooks and examples!

## Setting Up Environment Variables

PyRIT loads secrets and endpoints from environment variables or `.env` files. The `.env_example` file shows the format and available options.

### Environment Variable Precedence

When `initialize_pyrit_async` runs, environment variables are loaded in a specific order. **Later sources override earlier ones:**

```{mermaid}
flowchart LR
    A["1. System Environment"] --> B{"env_files provided?"}
    B -->|No| C["2. ~/.pyrit/.env"]
    C --> D["3. ~/.pyrit/.env.local"]
    B -->|Yes| E["2. Your specified files (in order)"]
```

**Default behavior** (no `env_files` argument):

| Priority | Source | Description |
|----------|--------|-------------|
| Lowest | System environment variables | Always loaded as the baseline |
| Medium | `~/.pyrit/.env` | Default config file (loaded if it exists) |
| Highest | `~/.pyrit/.env.local` | Local overrides (loaded if it exists) |

**Custom behavior** (with `env_files` argument): Only your specified files are loaded, in order. Default paths are completely ignored.

### Creating Your .env File

1. Copy `.env_example` to `.env` in your home directory in ~/.pyrit/.env
2. Add your API credentials. For example, for Azure OpenAI:

```bash
OPENAI_CHAT_ENDPOINT="https://your-resource.openai.azure.com/openai/deployments/your-deployment/chat/completions"
OPENAI_CHAT_KEY="your-api-key-here"
```

To find these values in Azure Portal: `Azure Portal > Azure AI Services > Azure OpenAI > Your OpenAI Resource > Resource Management > Keys and Endpoint`

### Using .env.local for Overrides

You can use `~/.pyrit/.env.local` to override values in `~/.pyrit/.env` without modifying the base file. This is useful for:
- Testing different targets
- Using personal credentials instead of shared ones
- Switching between configurations quickly

Simply create `.env.local` in your `~/.pyrit/` directory and add any variables you want to override.

### Custom Environment Files

You can also specify exactly which `.env` files to load using the `env_files` parameter:

```python
from pathlib import Path
from pyrit.setup import initialize_pyrit_async

await initialize_pyrit_async(
    memory_db_type="InMemory",
    env_files=[Path("./project-config.env"), Path("./local-overrides.env")]
)
```

When `env_files` is provided:
- **Only** the specified files are loaded (default paths are skipped entirely)
- Files are loaded in order—later files override earlier ones
- A `ValueError` is raised if any specified file doesn't exist

The CLI also supports custom environment files via the `--env-files` flag.

## Authentication Options

### API Keys (Default)
The simplest approach is using API keys as shown above. Most targets support this method.

### Azure Entra Authentication (Optional)
For Azure resources, you can use Entra auth instead of API keys. This requires:

1. Install Azure CLI for your OS:
   - [Windows](https://learn.microsoft.com/en-us/cli/azure/install-azure-cli-windows)
   - [Linux](https://learn.microsoft.com/en-us/cli/azure/install-azure-cli-linux)
   - [macOS](https://learn.microsoft.com/en-us/cli/azure/install-azure-cli-macos)

2. Log in to Azure:
   ```bash
   az login
   ```

When using Entra auth, you don't need to set API keys for Azure resources.

## Next Steps

- For detailed configuration options, see the [Configuration Guide](../code/setup/1_configuration.ipynb)
- For database options beyond in-memory storage, see the [Memory Documentation](../code/memory/0_memory.md)
