# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os

from pyrit.common import IN_MEMORY, initialize_pyrit

# This limits retries to 10 attempts with a 1 second wait between retries
# note this needs to be set before libraries that use them are imported

# Note this module needs to be imported at the top of a file so the values are modified

os.environ["RETRY_MAX_NUM_ATTEMPTS"] = "9"
os.environ["RETRY_WAIT_MIN_SECONDS"] = "0"
os.environ["RETRY_WAIT_MAX_SECONDS"] = "1"

initialize_pyrit(memory_db_type=IN_MEMORY)
