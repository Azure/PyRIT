# PyRIT Docker - Quick Start Guide

Docker container for PyRIT with support for both **Jupyter Notebook** and **GUI** modes.

## Prerequisites
- Docker installed and running
- `.env` file at `~/.pyrit/.env` with API keys
- Optionally, `~/.pyrit/.env.local` for additional environment variables

## Quick Start

### 1. Build the Image

Build from local source (includes frontend):
```bash
python docker/build_pyrit_docker.py --source local
```

Build from PyPI version:
```bash
python docker/build_pyrit_docker.py --source pypi --version 0.10.0
```

Rebuild base image (when devcontainer changes):
```bash
python docker/build_pyrit_docker.py --source local --rebuild-base
```

> **Note:** The build script automatically builds the devcontainer base image if needed.
> The base image is cached and reused for faster subsequent builds.

### 2. Run PyRIT

Jupyter mode (port 8888):
```bash
python docker/run_pyrit_docker.py jupyter
```

GUI mode (port 8000):
```bash
python docker/run_pyrit_docker.py gui
```

## Image Tags

Images are tagged with version information:
- PyPI: `pyrit:0.10.0`, `pyrit:latest`
- Local (clean): `pyrit:<full-commit-hash>`, `pyrit:latest`
- Local (modified): `pyrit:<full-commit-hash>-modified`, `pyrit:latest`

Run specific tag:
```bash
python docker/run_pyrit_docker.py gui --tag abc1234def5678
```

## Version Display

The GUI shows PyRIT version in a tooltip on the logo:
- PyPI builds: `0.10.0`
- Local builds: `abc1234def5678` or `abc1234def5678 + local changes`

## Docker Compose

Use profiles to run specific modes:

```bash
# Jupyter mode
docker-compose --profile jupyter up

# GUI mode
docker-compose --profile gui up
```

## Troubleshooting

**Image not found**: Run `python docker/build_pyrit_docker.py --source local` first

**.env missing**: Create `.env` file at `~/.pyrit/.env` with your API keys

**GUI frontend missing**: Build with `--source local` (PyPI builds before GUI release won't work)

For complete documentation, see [docker/README.md](./README.md)
