# %%NBQA-CELL-SEP52c935
from pyrit.common.apply_defaults import set_default_value
from pyrit.prompt_target import OpenAIChatTarget
from pyrit.setup.initializers.pyrit_initializer import PyRITInitializer


class CustomInitializer(PyRITInitializer):
    @property
    def name(self) -> str:
        return "Custom Configuration"
    
    @property
    def execution_order(self) -> int:
        return 2  # Lower numbers run first (default is 1)
    
    def initialize(self) -> None:
        set_default_value(class_type=OpenAIChatTarget, parameter_name="temperature", value=0.9)

    @property
    def description(self) -> str:
        return "Sets custom temperature for OpenAI targets"

# %%NBQA-CELL-SEP52c935
from pyrit.setup import initialize_pyrit
from pyrit.setup.initializers import SimpleInitializer

# Using built-in initializer
initialize_pyrit(
    memory_db_type="InMemory",
    initializers=[SimpleInitializer()]
)

# %%NBQA-CELL-SEP52c935
import os
import shutil
import tempfile

from pyrit.setup import initialize_pyrit

temp_dir = tempfile.mkdtemp()
script_path = os.path.join(temp_dir, "custom_init.py")

# This is the simple custom initializer from the "Creating an Initializer" section of this notebook
script_content = '''
from pyrit.setup.initializers.base import PyRITInitializer
from pyrit.common.apply_defaults import set_default_value
from pyrit.prompt_target import OpenAIChatTarget

class CustomInitializer(PyRITInitializer):
    @property
    def name(self) -> str:
        return "Custom Configuration"
    
    @property
    def execution_order(self) -> int:
        return 2  # Lower numbers run first (default is 1)
    
    def initialize(self) -> None:
        set_default_value(class_type=OpenAIChatTarget, parameter_name="temperature", value=0.9)

    @property
    def description(self) -> str:
        return "Sets custom temperature for OpenAI targets"

'''

with open(script_path, "w") as f:
    f.write(script_content)
    
print(f"Created: {script_path}")


initialize_pyrit(
    memory_db_type="InMemory",
    initialization_scripts=[temp_dir + "/custom_init.py"]
)


if os.path.exists(temp_dir):
    shutil.rmtree(temp_dir)
