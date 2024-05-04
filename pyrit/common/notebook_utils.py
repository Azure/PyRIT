# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

def is_in_ipython_session() -> bool:    
    try:
        __IPYTHON__
        return True
    except NameError:
        return False
