import warnings
import sys
import subprocess

# Suppress only the RuntimeWarning or all warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

print("Calling this")

# Execute jupytext
subprocess.run(["jupytext", "--execute", "--to", "notebook"] + sys.argv[1:])