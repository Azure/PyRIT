# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import shutil

from joblib import Memory
from logzero import logger


class CacheHelper(Memory):
    def __init__(self, shared_cache_dir: str = "", **kwargs):
        if shared_cache_dir and not os.path.exists(shared_cache_dir):
            raise IOError(f"{shared_cache_dir} does not exist")
        elif shared_cache_dir and not os.path.isdir(shared_cache_dir):
            raise IOError(f"{shared_cache_dir} is not a directory")
        self.shared_cache_file = os.path.join(shared_cache_dir, "embeddings_cache.zip") if shared_cache_dir else None
        super().__init__(**kwargs)

    def local_to_shared(self):
        if self.shared_cache_file and os.path.exists(self.location):
            logger.info(f"Downloading cache from {self.shared_cache_file} to {self.location}")
            shutil.make_archive(self.shared_cache_file[:-4], "zip", self.location)

    def shared_to_local(self):
        if self.shared_cache_file and os.path.exists(self.shared_cache_file):
            logger.info(f"Uploading cache from {self.location} to {self.shared_cache_file}")
            shutil.unpack_archive(self.shared_cache_file, self.location, "zip")
