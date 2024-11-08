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

```bash
pip install ipykernel
python -m ipykernel install --user --name=pyrit_kernel
```

Now you can start Jupyter Notebook:

```bash
jupyter notebook
```

Once the notebook is open, you can select the kernel that matches the name you gave earlier.
To do this, go to `Kernel > Change kernel > pyrit_kernel`.
