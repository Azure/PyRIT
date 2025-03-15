# PyRIT Docker Container

This Docker container provides a pre-configured environment for running PyRIT (Python Risk Identification Tool for generative AI) with JupyterLab integration.

## Features

- Pre-installed PyRIT with all dependencies
- JupyterLab integration for interactive usage
- CPU mode for broad compatibility
- Based on Microsoft Azure ML Python 3.12 inference image

## Directory Structure

```
.
├── Dockerfile
├── README.md
├── data/                  # Persistent data storage
├── docker-compose.yaml    # Docker Compose configuration
├── notebooks/             # Jupyter notebooks directory
│   └── example.ipynb      # Example PyRIT notebook
└── start.sh               # Container startup script
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

Once the container is running, you can access JupyterLab by opening a browser and navigating to:

```
http://localhost:8888
```

No password or token is required by default.

## Customization

### Adding Your Own Notebooks

Place your Jupyter notebooks in the `notebooks/` directory. They will be automatically available in JupyterLab.

### Adding Data

Place your datasets or other files in the `data/` directory. They will be accessible from the notebooks at `/app/data/`.

### Modifying the Configuration

Edit the `docker-compose.yaml` file to change port mappings, environment variables, or volume mounts.

## Using PyRIT in JupyterLab

```python
import pyrit
print(pyrit.__version__)

# Example PyRIT usage
# [Add specific examples based on PyRIT functionality]
```

## GPU Support (Optional)

This container is configured for CPU usage by default. To enable GPU support (requires NVIDIA drivers and container toolkit):

1. Edit `docker-compose.yaml`:
   ```yaml
   environment:
     - JUPYTER_ENABLE_LAB=yes
     - ENABLE_GPU=true  # Add this line
   
   # Add this section
   deploy:
     resources:
       reservations:
         devices:
           - driver: nvidia
             count: all
             capabilities: [gpu]
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

If you encounter permission issues with notebooks or data, ensure the directories have appropriate permissions:

```bash
chmod -R 777 notebooks/ data/
```


## Version Information

- Base Image: `mcr.microsoft.com/azureml/minimal-py312-inference:latest`
- Python: 3.12
- PyTorch: Latest version with CUDA support
- Pyrit: Latest version from main branch

## Customization

You can customize the container by:

1. Modifying the Dockerfile to include additional dependencies
2. Adding your own notebooks to the `/app/notebooks` directory
3. Changing the startup options in `start.sh`

## Security Note

The JupyterLab instance runs without authentication by default for ease of use. For production deployments, consider adding authentication or running behind a secured proxy.y

