# FAQs

## I have an API endpoint. How can I use PyRIT to connect to it?

To connect to your endpoint you need a client that knows how to work with that type of endpoint.
Make sure to check the `pyrit.chat` [module](https://github.com/Azure/PyRIT/tree/main/pyrit/chat) for existing clients such as `AzureOpenAIChat`.
If no corresponding class exists, you can write your own and implement the `ChatSupport` interface.
If you do this please consider contributing your new client back to PyRIT.

## What can I do if Jupyter cannot find PyRIT?

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

Now your Jupyter Notebook should be able to find and import the libraries installed in your conda environment.

## What if you are using a Windows 11 operating system and having issues installing PyRIT. 
When running 'pip install pyrit' you may run into the long name path issue documented here: https://learn.microsoft.com/en-us/windows/win32/fileio/maximum-file-path-limitation?tabs=powershell#enable-long-paths-in-windows-10-version-1607-and-later

To resolve this via Powershell run the following with administrative permissions:
```Powershell
New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" `
-Name "LongPathsEnabled" -Value 1 -PropertyType DWORD -Force
```
