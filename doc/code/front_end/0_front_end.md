# PyRIT Command-Line Frontends

PyRIT provides two command-line interfaces for running AI red teaming scenarios:

## pyrit_scan - Single-Command Execution

`pyrit_scan` is designed for **automated, non-interactive scenario execution**. It's ideal for:
- CI/CD pipelines and automated testing workflows
- Batch processing multiple scenarios with scripts
- One-time security assessments
- Reproducible test runs with exact parameters

Each invocation runs a single scenario with specified parameters and exits, making it perfect for automation where you need clean, scriptable execution with predictable exit codes.

**Key characteristics:**
- Loads PyRIT modules fresh for each execution
- Runs one scenario per command
- Exits with status code (0 for success, non-zero for errors)
- Output can be easily captured and parsed

**Documentation:** [1_pyrit_scan.ipynb](1_pyrit_scan.ipynb)

## pyrit_shell - Interactive Session

`pyrit_shell` is an **interactive REPL (Read-Eval-Print Loop)** for exploratory testing. It's ideal for:
- Interactive scenario development and debugging
- Rapid iteration and experimentation
- Exploring multiple scenarios without reload overhead
- Session-based result tracking and comparison

The shell loads PyRIT modules once at startup and maintains a persistent session, allowing you to run multiple scenarios quickly and review their results interactively.

**Key characteristics:**
- Fast subsequent executions (modules loaded once)
- Session history of all runs
- Interactive result exploration and printing
- Persistent context across multiple scenario runs
- Tab completion and command help

**Documentation:** [2_pyrit_shell.md](2_pyrit_shell.md)
