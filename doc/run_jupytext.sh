#!/bin/bash

curr_dir="$(dirname "$(realpath "$0")")"
find "$curr_dir" \( -type d -name "deployment" -prune \) -o \( -type f -name "*.py" -print0 \) | while IFS= read -r -d '' file
do
    echo "Processing $file"
    jupytext --execute --to notebook "$file"
done
