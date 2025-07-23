# What can I do if Jupyter cannot find PyRIT?

First, you need to find the corresponding conda environment for your project.
You can do this with the following command:

```bash
conda env list
```

Then activate it using

```bash
conda activate <env_name>
```

Next, you need to install the IPython kernel in the virtual environment.

Note: Jupyter and ipykernel are no longer installed by default with the base package. If you need to use Jupyter notebooks with PyRIT, you'll need to install these dependencies using one of the following methods:

1. Install with development dependencies: `pip install -e ".[dev]"`
2. Install with all optional dependencies: `pip install -e ".[all]"`
3. Install just the notebook dependencies manually: `pip install jupyter ipykernel`

After installing these dependencies, you can proceed with the kernel setup steps below.

```bash
python -m ipykernel install --user --name=pyrit_kernel
```

Now you can start Jupyter Notebook:

```bash
jupyter notebook
```

Once the notebook is open, you can select the kernel that matches the name you gave earlier.
To do this, go to `Kernel > Change kernel > pyrit_kernel`.
