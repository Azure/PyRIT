# 9. Pre-Commit Hooks

There are a number of pre-commit hooks available to run on files within the repo. Run these once you have code that you'd like to submit in a pull request to make sure they pass. These are meant to enforce style within the code base.

```bash
### Make sure all files are added with `git add` before running pre-commit

# run hooks on all files
pre-commit run --all-files

# run hooks on a specific file
pre-commit run --files <file_name>
