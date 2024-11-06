# Install PyRIT

To install PyRIT using pip, make sure you have Python 3.10, 3.11, or 3.12 installed using `python --version`.
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

Note that notebooks and your PyRIT installation need to be on the same version.
If you install PyRIT from source then the notebooks from the same cloned
repository will work. If you install PyRIT from PyPI (like in the instructions
above) then you'll need to download the notebook from the corresponding
release branch. For example, if you installed `pyrit==0.2.1` then the
corresponding notebooks will be at
https://github.com/Azure/PyRIT/tree/releases/v0.2.1/doc

To check your PyRIT version run `pip freeze` from a terminal or

```python
import pyrit
pyrit.__version__
```
in the Python REPL.

## Other Resources

- To install as a contributor, see [contributor installation](../contributing/installation.md)
- To populate secrets, see [populating secrets](./populating_secrets.md)
