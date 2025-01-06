# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
import sys

from pyrit.common.path import LOG_PATH

fmt = "[%(asctime)s][%(msecs)d][%(name)s][%(levelname)s][%(message)s]"
log_formatter = logging.Formatter(fmt=fmt, datefmt="%H:%M:%S")

file_handler = logging.FileHandler(filename=LOG_PATH, mode="a+")
file_handler.setFormatter(log_formatter)
file_handler.setLevel(logging.INFO)

console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(log_formatter)
console_handler.setLevel(logging.INFO)

logger = logging.getLogger("ai-red-team")
logger.addHandler(file_handler)
logger.addHandler(console_handler)
logger.setLevel(logging.INFO)
