# Install PyRIT with Conda

For a traditional Python installation on your local machine, follow the instructions below.

**Use local installation if you:**
- Want to integrate PyRIT into existing Python workflows
- Prefer a lighter-weight installation
- Need to customize your Python environment
- Want direct access to PyRIT from your system Python

## Local Installation Instructions

To install PyRIT using pip, make sure you have Python 3.10, 3.11, 3.12, or 3.13 installed using `python --version`.
Alternatively, create a conda environment as follows

```
conda create -y -n <environment-name> python=3.11
```

followed by `conda activate <environment-name>`

Once the environment with the correct Python version is set up, run

```
pip install pyrit
```

Next, check out our [docs](../index.md) and run the notebooks in your environment!

```{important}
**Matching Notebooks to Your Version:**

Notebooks and your PyRIT installation must be on the same version. This pip installation gives you the **latest stable release** from PyPI.

1. **Check your installed version:**
   ```bash
   pip freeze | grep pyrit
   ```

   Or in Python:
   ```python
   import pyrit
   print(pyrit.__version__)
   ```

3. **Match notebooks to your version**:
   - If using a **release version** (e.g., `0.9.0`), download notebooks from the corresponding release branch: `https://github.com/Azure/PyRIT/tree/releases/v0.9.0/doc`
   - The automatically cloned notebooks from the `main` branch may not match your installed version
   - This website documentation shows the latest development version (`main` branch).

3. **If you installed from source:** The notebooks in your cloned repository will already match your code version.
