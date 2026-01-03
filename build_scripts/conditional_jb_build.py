# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Conditional Jupyter Book build wrapper for pre-commit.

This script checks the RUN_LONG_PRECOMMIT environment variable:
- If set to "true", runs the full `jb build -W -q ./doc` command
- Otherwise, exits successfully (fast validation script runs instead)

This allows CI/pipeline to run full builds while local development uses fast validation.
"""

import os
import subprocess
import sys


def main():
    run_long = os.environ.get("RUN_LONG_PRECOMMIT", "").lower() == "true"

    if run_long:
        print("RUN_LONG_PRECOMMIT=true: Running full Jupyter Book build...")
        # Run jb build with the same flags as before
        result = subprocess.run(
            ["jb", "build", "-W", "-q", "./doc"],
            cwd=os.path.dirname(os.path.dirname(__file__)),  # Repository root
        )
        return result.returncode
    else:
        print("RUN_LONG_PRECOMMIT not set: Skipping full Jupyter Book build (fast validation runs instead)")
        return 0


if __name__ == "__main__":
    sys.exit(main())
