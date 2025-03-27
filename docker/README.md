# PyRIT Docker Container

This Docker container provides a pre-configured environment for running PyRIT (Python Risk Identification Tool for generative AI) with JupyterLab integration. It comes with pre-installed PyRIT, all necessary dependencies, and supports both CPU and optional GPU modes.

## Features

- Pre-installed PyRIT with all dependencies
- JupyterLab integration for interactive usage
- CPU mode enabled by default for broad compatibility
- Option to enable GPU support (requires NVIDIA drivers and container toolkit)
- Automatic documentation cloning from the PyRIT repository when `CLONE_DOCS=true`
- Based on Microsoft Azure ML Python 3.12 inference image

## Directory Structure

```
.
├── Dockerfile                       # Container build configuration
├── README.md                        # This documentation file
├── requirements.txt                 # Python packages
├── docker-compose.yaml              # Docker Compose configuration
├── .env_container_settings_example  # Container's example env file (rename .env.container.settings)
└── start.sh                         # Container startup script
```

## Prerequisites

- [Docker](https://docs.docker.com/get-docker/)
- [Docker Compose](https://docs.docker.com/compose/install/)

## Quick Start

### Building and Running the Container

```bash
# Build and start the container in detached mode
docker-compose up -d

# View logs
docker-compose logs -f

# Stop the container
docker-compose down
```

### Accessing JupyterLab

Once the container is running, open your browser and navigate to:

```
http://localhost:8888
```

By default, JupyterLab is configured to run without a password or token.

## Configuration

### Environment Variables

- **CLONE_DOCS**: When set to `true` (default), the container automatically clones the PyRIT repository and copies the documentation files to the notebooks directory. To disable this behavior, set `CLONE_DOCS=false` in your environment or in the `.env.container.settings` file.
- **ENABLE_GPU**: Set to `true` to enable GPU support (requires NVIDIA drivers and container toolkit). The container defaults to CPU-only mode.

The container expects a .env file and optionally a .env.local file to provide secret keys and configuration values. If these files do not exist, please create them by copying the provided example files:
```bash
cp ../.env.example ../.env
cp ../.env.local_example ../.env.local
cp .env_container_settings_example .env.container.settings
```
These files will automatically be pulled into the container (.env and .env.local), and if they're missing, you might encounter errors indicating that required environment files are not found.


### Adding Your Own Notebooks and Data

- **Notebooks**: Place your Jupyter notebooks in the `notebooks/` directory. They will be available automatically in JupyterLab.
- **Data**: Place your datasets or other files in the `data/` directory. Access them from your notebooks at `/app/data/`.

### Important Permission Configuration

Ensure your `notebooks/` , `data/` and `../assets/` directories have the correct permissions to allow container access:

```bash
chmod -R 777 notebooks/ data/ ../assets
```

## Recommended Docker Compose Configuration

To correctly map your local notebooks and data directories into the container, use the following Docker Compose configuration:

```yaml
services:
  pyrit:
    build:
      context: .
      dockerfile: Dockerfile
    image: pyrit:latest
    container_name: pyrit-jupyter
    ports:
      - "8888:8888"
    volumes:
      - ./notebooks:/app/notebooks
      - ./data:/app/data
      - ../assets:/app/assets
    env_file:
      - ../.env
      - ../.env.local
      - .env.container.settings
    restart: unless-stopped
    healthcheck:
      test: ["CMD-SHELL", "curl -sf http://localhost:8888 || exit 1"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s

volumes:
  notebooks:
  data:
```

## Modifying the Configuration

Edit the `docker-compose.yaml` file to change port mappings, environment variables, or volume mounts as needed.

## Using PyRIT in JupyterLab

Start a new notebook in JupyterLab and try the following:

```python
import pyrit
print(pyrit.__version__)

# Example PyRIT usage:
# [Insert your PyRIT usage examples here]
```

## GPU Support (Optional)

To enable GPU support:

1. Edit `.env.container.settings` and add/modify the following:

   ```bash
    ENABLE_GPU=true  # Enable GPU support
   ```

2. Restart the container:

   ```bash
   docker-compose down
   docker-compose up -d
   ```

## Troubleshooting

### JupyterLab Not Accessible

If you cannot access JupyterLab, check the container logs:

```bash
docker-compose logs pyrit
```

### Permission Issues

If you encounter permission issues with the notebooks or data directories, adjust the permissions:

```bash
chmod -R 777 notebooks/ data/ ../assets/
```

## Version Information

- **Base Image**: `mcr.microsoft.com/azureml/minimal-py312-inference:latest`
- **Python**: 3.12
- **PyTorch**: Latest version with CUDA support
- **PyRIT**: Installed from PyPI (latest version)

## Customization

You can further customize the container by:

1. Modifying the `Dockerfile` to add additional system or Python dependencies.
2. Adding your own notebooks to the `/app/notebooks` directory.
3. Changing startup options in the `start.sh` script.

## Security Note

The JupyterLab instance is configured to run without authentication by default for ease of use. For production deployments, consider adding authentication or running behind a secured proxy.
