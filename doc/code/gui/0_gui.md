# PyRIT GUI (CoPyRIT)

CoPyRIT is a web-based graphical interface for PyRIT built with React and Fluent UI. It provides an interactive way to run attacks, configure targets and converters, and view results — all from a browser.

> **Note:** The older Gradio-based GUI (`HumanInTheLoopScorerGradio`) and the `HumanInTheLoopConverter` are deprecated and will be removed in v0.13.0. CoPyRIT covers a much broader part of the user journey — from attack creation and converter configuration to attack history and result analysis — making these limited interactive components obsolete.

## Getting Started

There are several ways to run CoPyRIT:

### PyRIT Backend CLI

If you have PyRIT installed, use the `pyrit_backend` command to start the server. The bundled frontend is served automatically.

```bash
pyrit_backend
```

Then open `http://localhost:8000` in your browser.

### Docker

CoPyRIT is also available as a Docker container. See the [Docker setup](https://github.com/Azure/PyRIT/blob/main/docker/) for details.

### Azure Deployment

Azure-hosted deployment is planned for the near future.
