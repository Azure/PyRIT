# Find all .py files in the docs directory as this script and convert them to .ipynb
# This excludes the deployment directory

import argparse
import os
import subprocess
from pathlib import Path

skip_files = {
    "conf.py",
    "0_auxiliary_attacks.py",
    "1_gcg_azure_ml.py",
    "6_human_converter.py",
    "HITL_Scoring_Orchestrator.py",
}

exec_dir = Path(os.getcwd())
file_type = ".py"
excluded_dir = {"deployment", "generate_docs"}
cache_path = os.path.join(exec_dir, "cache")
kernel_name = "pyrit_kernel"


def main():
    parser = argparse.ArgumentParser(description="Converts .py files in docs to .ipynb")
    parser.add_argument("-id", "--run_id", type=str, help="id used to cache processed files")
    parser.add_argument(
        "-kn",
        "--kernel_name",
        default=kernel_name,
        type=str,
        help=f"name of kernel to run notebooks. (default: {kernel_name})",
    )
    args = parser.parse_args()

    cache_file = os.path.join(cache_path, args.run_id)
    processed_files = set()
    if os.path.isfile(cache_file):
        with open(cache_file, "r") as f:
            for file_path in f:
                processed_files.add(file_path)

    found_files = find_files(exec_dir, file_type)

    for file in found_files:
        if file in processed_files:
            print(f"Skipping already processed file: {file}")
            continue
        if any([skip_file in file for skip_file in skip_files]):
            print(f"Skipping configured skipped file: {file}")
            continue
        print(f"Processing {file}")
        result = subprocess.run(
            ["jupytext", "--execute", "--set-kernel", args.kernel_name, "--to", "notebook", file],
            stdout=subprocess.DEVNULL,
        )
        if result.returncode != 0:
            print(f"Error processing {file}")
            continue
        print(f"Successfully processed {file}")
        # Log to cache file
        f = open(cache_file, "a")
        f.write(os.path.join(file))
        f.close()
    return


def find_files(directory, file_extension):
    matches = []
    for root, dirs, files in os.walk(directory):
        for dir in excluded_dir:
            if dir in dirs:
                dirs.remove(dir)
        for file in files:
            if file.endswith("_helpers.py"):
                continue
            if file.endswith(file_extension):
                matches.append(os.path.join(root, file))
    return matches


if __name__ == "__main__":
    main()
