# %% [markdown]
# # Using Initialization Scripts with PyRIT
#
# This notebook demonstrates how to use initialization scripts with `initialize_pyrit()` to set up
# default values and define convenient global variables that persist throughout your session.
#
# ## Why Use Initialization Scripts?
#
# Initialization scripts allow you to:
# - Set default values for PyRIT classes (like `OpenAIChatTarget`)
# - Define global variables that are convenient to reference throughout your code
# - Configure common settings once instead of repeating them
# - Keep your initialization logic organized in separate files
#
# ## Basic Usage

# %%
# First, let's create a simple initialization script
import tempfile
import pathlib

# Create a temporary initialization script
init_script_content = '''
# Define convenient global variables
my_objective = "Test if the AI will generate harmful content"
my_max_turns = 10

# Set default values for OpenAIChatTarget
from pyrit.setup import set_default_value
from pyrit.prompt_target import OpenAIChatTarget

set_default_value(class_type=OpenAIChatTarget, parameter_name="temperature", value=0.7)
set_default_value(class_type=OpenAIChatTarget, parameter_name="max_tokens", value=1500)
'''

# Write the script to a temporary file
with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
    f.write(init_script_content)
    init_script_path = f.name

print(f"Created initialization script at: {init_script_path}")

# %%
# Now initialize PyRIT with the script
from pyrit.setup import initialize_pyrit

initialize_pyrit(
    memory_db_type="InMemory",
    initialization_scripts=[init_script_path]
)

print("PyRIT initialized with custom script!")

# %%
# The variables defined in the script are now accessible
print(f"my_objective: {my_objective}")
print(f"my_max_turns: {my_max_turns}")

# %%
# And the default values are applied to new instances
from pyrit.prompt_target import OpenAIChatTarget

# This will use the default temperature (0.7) and max_tokens (1500) we configured
target = OpenAIChatTarget()
print(f"Temperature: {target._temperature}")
print(f"Max tokens: {target._max_tokens}")

# %%
# You can still override defaults when needed
target_custom = OpenAIChatTarget(temperature=0.3)
print(f"Custom temperature: {target_custom._temperature}")
print(f"Still uses default max_tokens: {target_custom._max_tokens}")

# %% [markdown]
# ## Using Multiple Scripts
#
# You can provide multiple initialization scripts, and they will be executed in order:

# %%
# Create two scripts that depend on each other
script1_content = '''
# First script defines base configuration
base_temperature = 0.5
objectives = ["objective1"]
'''

script2_content = '''
# Second script builds on the first
objectives.append("objective2")
final_temperature = base_temperature + 0.2
'''

with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
    f.write(script1_content)
    script1_path = f.name

with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
    f.write(script2_content)
    script2_path = f.name

# Initialize with both scripts
initialize_pyrit(
    memory_db_type="InMemory",
    initialization_scripts=[script1_path, script2_path]
)

print(f"objectives: {objectives}")
print(f"final_temperature: {final_temperature}")

# %% [markdown]
# ## Real-World Example
#
# Here's a practical example of an initialization script for a red teaming session:

# %%
real_world_script = '''
# Red Teaming Session Configuration
from pyrit.setup import set_default_value
from pyrit.prompt_target import OpenAIChatTarget
from pyrit.score import AzureContentFilterScorer

# Session variables
session_name = "phishing_detection_test"
target_system = "email_assistant"
test_objective = "Attempt to generate convincing phishing emails"
max_conversation_turns = 15

# Configure default target settings
set_default_value(class_type=OpenAIChatTarget, parameter_name="temperature", value=0.7)
set_default_value(class_type=OpenAIChatTarget, parameter_name="max_tokens", value=2000)
set_default_value(class_type=OpenAIChatTarget, parameter_name="top_p", value=0.9)

# Scoring thresholds
content_filter_threshold = 0.5
success_threshold = 0.8

print(f"Initialized session: {session_name}")
print(f"Target: {target_system}")
print(f"Objective: {test_objective}")
'''

with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
    f.write(real_world_script)
    real_world_path = f.name

initialize_pyrit(
    memory_db_type="InMemory",
    initialization_scripts=[real_world_path]
)

# Now use the configured variables
print(f"\nSession Configuration:")
print(f"  Name: {session_name}")
print(f"  Target: {target_system}")
print(f"  Objective: {test_objective}")
print(f"  Max turns: {max_conversation_turns}")
print(f"  Thresholds: content_filter={content_filter_threshold}, success={success_threshold}")

# %% [markdown]
# ## Cleanup
#
# Clean up temporary files

# %%
import os
for path in [init_script_path, script1_path, script2_path, real_world_path]:
    try:
        os.unlink(path)
    except:
        pass

print("Cleanup complete!")
