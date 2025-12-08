# Install PyRIT with Docker

Docker provides the fastest way to get started with PyRIT. This method uses a pre-configured container that can run either a Web GUI or JupyterLab, eliminating the need for local Python environment setup.

## Who Should Use Docker?

✅ **Use Docker if you:**
- Want to get started immediately without Python setup
- Prefer a consistent, isolated environment
- Are new to PyRIT and want to try it quickly
- Want Web GUI or JupyterLab pre-configured and ready to go
- Work on Windows, macOS, or Linux

❌ **Consider [local installation](./1a_install_conda.md) if you:**
- Need to integrate PyRIT into existing Python workflows
- Prefer lighter-weight installations
- Want direct access to PyRIT from your system Python

```{important}
**Version Compatibility:** This Docker setup can install either the **latest stable release** from PyPI or build from **local source**. When using a PyPI release, notebooks and documentation must match your PyRIT version. Download notebooks from the corresponding release branch if using a stable release.
```

## Prerequisites

Before starting, install:

- [Docker](https://docs.docker.com/get-docker/)

```{note}
On Windows, we recommend Docker Desktop. On Linux, you can install Docker Engine directly.
```

## Quick Start

### 1. Clone the PyRIT Repository

```bash
git clone https://github.com/Azure/PyRIT
cd PyRIT
```

### 2. Set Up Environment Files

Create your environment configuration:

```bash
# Create main environment files
cp .env_example .env
cp .env_local_example .env.local
```

```{important}
Edit the `.env` and `.env.local` files to add your API keys and configuration values. See [populating secrets](./populating_secrets.md) for details.
```

### 3. Build the Docker Image

Choose between PyPI release or local source:

```bash
# Option A: Build from local source (includes GUI)
python docker/build_pyrit_docker.py --source local

# Option B: Build from PyPI release (Jupyter only until GUI is released)
python docker/build_pyrit_docker.py --source pypi --version 0.10.0
```

### 4. Run PyRIT

Choose your preferred mode:

#### Web GUI Mode (Port 8000)

Run the interactive web interface:

```bash
python docker/run_pyrit_docker.py gui
```

Then open your browser to: **http://localhost:8000**

The GUI provides:
- Interactive chat interface
- Target selection and configuration
- Prompt converter management
- Real-time conversation history

#### Jupyter Mode (Port 8888)

Run JupyterLab for notebook-based exploration:

```bash
python docker/run_pyrit_docker.py jupyter
```

Then open your browser to: **http://localhost:8888**

By default, JupyterLab runs without authentication for ease of use.

```{warning}
The default configuration has no password. For production use, consider adding authentication.
```

## Using PyRIT

### In Web GUI Mode

1. **Select a target** from the sidebar (e.g., Azure OpenAI)
2. **Configure the target** with your endpoint and deployment details
3. **Start chatting** - send prompts and see responses
4. **Apply converters** to transform your prompts before sending

### In JupyterLab Mode

Once JupyterLab is open:

1. **Navigate to the notebooks**: The PyRIT documentation notebooks will be automatically available in the `notebooks/` directory
2. **Check your PyRIT version**:

```python
import pyrit
print(pyrit.__version__)
```

3. **Match notebooks to your version**:
   - If using PyPI with a **release version** (e.g., `0.10.0`), ensure your local branch matches: `git checkout releases/v0.10.0`
   - Notebooks in the container come from your local branch at build time
   - For local source builds, notebooks match your current branch

4. **Start using PyRIT**:

```python
# Your PyRIT code here
```

Check out the [cookbooks](../cookbooks/README.md) for example workflows and tutorials.

## Advanced Usage

### Docker Compose (Alternative)

You can also use docker-compose with profiles:

```bash
# Run Jupyter mode
cd docker
docker-compose --profile jupyter up

# Run GUI mode
docker-compose --profile gui up
```

### Specific Image Tags

Run a specific version:

```bash
# List available images
docker images pyrit

# Run specific tag
python docker/run_pyrit_docker.py gui --tag abc1234def5678
```

### Version Information

In GUI mode, hover over the logo to see the PyRIT version. This shows:
- PyPI releases: `0.10.0`
- Local builds: Full commit hash or `<commit> + local changes`

## Configuration Options

### Environment Variables

The container includes these defaults (can be overridden in `.env` or `.env.local`):

- **ENABLE_GPU**: Set to `true` to enable GPU support (requires NVIDIA drivers and container toolkit)
- **PYRIT_MODE**: Automatically set by the run script (`jupyter` or `gui`)

## Container Management

### Stop the Container

```bash
# If using run script (Ctrl+C in the terminal)
# Or force stop:
docker stop pyrit-jupyter
docker stop pyrit-gui
```

### View Running Containers

```bash
docker ps
```

### View Logs

```bash
docker logs pyrit-jupyter -f
docker logs pyrit-gui -f
```

### Rebuild After Changes

```bash
python docker/build_pyrit_docker.py --source local
```

## GPU Support (Optional)

To use NVIDIA GPUs with PyRIT:

### Prerequisites

1. Install [NVIDIA drivers](https://www.nvidia.com/Download/index.aspx)
2. Install [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)

### Enable GPU in Container

1. Edit `.env.container.settings`:

   ```bash
   ENABLE_GPU=true
   ```

2. Restart the container:

   ```bash
   docker-compose down
   docker-compose up -d
   ```

3. Verify GPU access in a notebook:

   ```python
   import torch
   print(f"CUDA available: {torch.cuda.is_available()}")
   print(f"GPU count: {torch.cuda.device_count()}")
   ```

## Troubleshooting

### JupyterLab Not Accessible

**Problem**: Cannot access `http://localhost:8888`

**Solutions**:
1. Check if the container is running:
   ```bash
   docker ps
   ```

2. View container logs:
   ```bash
   docker-compose logs pyrit
   ```

3. Ensure port 8888 is not already in use:
   ```bash
   # On Linux/macOS
   lsof -i :8888

   # On Windows (PowerShell)
   netstat -ano | findstr :8888
   ```

### Permission Errors

**Problem**: Permission denied errors when accessing notebooks or data

**Solution**: Set appropriate permissions:

```bash
chmod -R 777 notebooks/ data/ ../assets/
```

### Missing Environment Files

**Problem**: Container fails with missing environment file errors

**Solution**: Ensure all environment files are created:

```bash
ls -la ../.env ../.env.local .env.container.settings
```

If any are missing, create them from the examples as shown in step 2 of Quick Start.

### Container Build Fails

**Problem**: Docker build fails with dependency errors

**Solutions**:
1. Clear Docker cache and rebuild:
   ```bash
   docker-compose build --no-cache
   ```

2. Ensure you have sufficient disk space:
   ```bash
   docker system df
   ```

3. Prune old images if needed:
   ```bash
   docker system prune -a
   ```

### Image Not Found

**Problem**: `ERROR: Docker image 'pyrit:latest' not found!`

**Solution**: Build the image first:

```bash
python docker/build_pyrit_docker.py --source local
```

### Frontend Not Found (GUI Mode)

**Problem**: GUI mode shows "Frontend not found" error

**Solution**: The GUI is only available when building from local source (PyPI builds before the GUI release don't include it):

```bash
python docker/build_pyrit_docker.py --source local
python docker/run_pyrit_docker.py gui
```

### PYRIT_MODE Not Set

**Problem**: Container exits with "PYRIT_MODE environment variable is not set"

**Solution**: Always use the run scripts instead of running docker directly:

```bash
python docker/run_pyrit_docker.py jupyter
# or
python docker/run_pyrit_docker.py gui
```

## Additional Resources

- [docker/QUICKSTART.md](https://github.com/Azure/PyRIT/blob/main/docker/QUICKSTART.md) - Quick reference guide
- [Populating Secrets](./populating_secrets.md) - How to configure API keys
- [Cookbooks](../cookbooks/README.md) - Example workflows and tutorials
