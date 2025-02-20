# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import json
import os
import sys

def remove_kernelspec_from_ipynb_files(file_path: str):
    # Iterate through all .ipynb files in the specified directory and its subdirectories
    if file_path.endswith(".ipynb"):
        print('file: ')
        print(file_path)
        with open(file_path, "r", encoding="utf-8") as f:
            content = json.load(f)

            # Remove the "kernelspec" metadata section if it exists
            if "metadata" in content and "kernelspec" in content["metadata"]:
                del content["metadata"]["kernelspec"]
                print(f"Removed kernelspec from {file_path}")
            # Save the modified content back to the .ipynb file
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(content, f, indent=2)

if __name__ == "__main__":
    files = sys.argv[1:]
    
    # # Remove the "kernelspec" section from all .ipynb files in the specified directory
    for file in files:
        remove_kernelspec_from_ipynb_files(file)
    print("DONE")
    
