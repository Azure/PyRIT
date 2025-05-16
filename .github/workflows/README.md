# PyPI Publishing Workflow

This GitHub workflow automates the process of publishing the pyrit-sqlalchemy14 package to PyPI.

## How It Works

The workflow is triggered in two ways:
1. When you push a new tag that starts with 'v' (e.g., v0.9.1)
2. Manually through the GitHub Actions interface (workflow_dispatch)

## Setting Up the PyPI API Token

For this workflow to function properly, you need to add your PyPI API token as a GitHub secret:

1. Go to your GitHub repository
2. Navigate to Settings > Secrets and variables > Actions
3. Click on "New repository secret"
4. Name: `PYPI_API_TOKEN`
5. Value: Your PyPI API token
   
## Usage

### Publishing via Tags

To publish a new version:

1. Update the version in both pyproject.toml and pyproject.poetry.toml
2. Commit the changes
3. Create and push a new tag:
   ```
   git tag v0.9.1
   git push origin v0.9.1
   ```

### Manual Publishing

You can also trigger the workflow manually from the Actions tab in your GitHub repository.
