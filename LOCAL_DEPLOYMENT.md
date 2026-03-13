# Local Deployment Guide

This fork keeps the core Azure PyRIT project, but the day-to-day local setup is a little different.

Use this guide for the version in this repository. Keep using the official Azure docs for the broader PyRIT concepts, notebooks, and full platform support.

## What Is Different In This Fork

This local version is centered around a simpler prompt-builder workflow instead of the older chat-first UI.

Main local characteristics:

- The frontend is a prompt builder, not a chat shell.
- The left panel surfaces all available converter options by default.
- The builder is biased toward video-generator testing, while still exposing the full PyRIT converter list.
- Grok is used as the helper model that rewrites or expands prompts. It is not the system under test unless you explicitly choose to test Grok.
- Some rewrite-style options include a `Prompt length` control so Grok can return a longer version of the prompt when that makes sense.
- The builder can work without a concrete target selected. This is useful when you are building prompts before deciding where to run them.
- For file-based flows, the current UI expects an existing file path or URL. It does not yet upload a new browser file directly from the builder.

## How The Local Setup Fits Together

There are three practical ways to use this repo locally.

### 1. Docker Jupyter

Best when you want the classic PyRIT notebook workflow.

- Runs a single Docker container on port `8888`
- Opens JupyterLab
- Uses your local secrets from `~/.pyrit/.env` and `~/.pyrit/.env.local`

This is the safest option if you want to avoid extra local processes.

### 2. Local Prompt Builder

Best when you want the new UI in this fork.

- Runs the FastAPI backend locally
- Runs the Vite frontend locally
- Uses Grok as the helper model for prompt rewrites and previews
- Lets you browse converters, fill settings, and preview the output

This is the setup that matches the current product direction of this fork.

### 3. Default Frontend Dev Mode

Best when you are changing frontend code and do not need the Grok-backed prompt-builder behavior.

- Uses the existing `frontend/dev.py` helper
- Starts frontend on `3000`
- Starts backend on `8000`
- Good for general frontend work
- Not the recommended path if you specifically want the Grok-backed builder behavior

## Secrets And Environment Files

PyRIT already knows to look for these files:

- `~/.pyrit/.env`
- `~/.pyrit/.env.local`

Later files win, so `~/.pyrit/.env.local` is the right place for your real local secrets.

For this fork, the minimum useful Grok setup is:

```env
OPENAI_CHAT_ENDPOINT="https://api.x.ai/v1"
OPENAI_CHAT_KEY="YOUR_GROK_KEY"
OPENAI_CHAT_MODEL="grok-4-latest"
OPENAI_CHAT_UNDERLYING_MODEL="grok-4-latest"
```

Those values make Grok available as the helper model for converter previews and prompt rewrites.

## Recommended Local Run Modes

### Option A: Run Jupyter In Docker

Use this when you want notebooks and want to keep Docker usage simple.

From the repo root:

```bash
docker build -f .devcontainer/Dockerfile -t pyrit-devcontainer .devcontainer
cp docker/.env_container_settings_example docker/.env.container.settings
docker compose -f docker/docker-compose.yaml --profile jupyter up --build
```

Open:

```text
http://127.0.0.1:8888/lab
```

Important local note:

- `pyrit-jupyter` is the only PyRIT container you need for notebooks.
- Do not also start the `gui` Docker profile unless you actually want a second PyRIT container.

### Option B: Run The Grok-Backed Prompt Builder

Use this when you want the current local UI.

Start the backend from the repo root:

```bash
PYRIT_CORS_ORIGINS="http://127.0.0.1:4174,http://localhost:4174,http://127.0.0.1:3000,http://localhost:3000" \
pyrit_backend --host 127.0.0.1 --port 8002 --initialization-scripts scripts/grok_builder_initializer.py
```

Start the frontend in a second terminal:

```bash
cd frontend
VITE_API_URL="http://127.0.0.1:8002/api" npm run dev -- --host 127.0.0.1 --port 4174
```

Open:

```text
http://127.0.0.1:4174/
```

What this mode gives you:

- All converter options visible in the left panel
- Video-oriented grouping to help people testing video generators
- Grok-backed previews for rewrite-style options
- Prompt-length controls where longer output makes sense

### Option C: Run The Default Dev Setup

Use this when you just want the stock local frontend dev flow.

From `frontend/`:

```bash
npm install
python dev.py start
```

Open:

```text
http://127.0.0.1:3000/
```

API docs:

```text
http://127.0.0.1:8000/docs
```

Use this mode if you are doing normal UI work. Use the Grok-backed mode above if you want the prompt-builder previews to behave like this fork expects.

## How The Prompt Builder Works

The current builder is a thin layer over PyRIT converters.

The flow is:

1. The frontend asks the backend for the available converter types.
2. The backend returns the converter settings and lightweight UI metadata.
3. The user picks a converter and fills in the settings.
4. For rewrite-style converters, the backend sends that request through Grok.
5. The frontend shows the transformed output and explains what changed.

This means the builder is not inventing a new attack engine. It is mostly making the existing PyRIT converter system easier to browse and use.

## Local Behavior To Know About

These are the practical rules for this fork right now:

- Grok is the helper model for building and previewing prompts.
- The target you are testing can be left blank while building prompts.
- Some converters still need extra setup, and the UI should say that clearly.
- Some converters are better for long prompts, and the UI marks those.
- `Prompt length` is a soft instruction, not an exact character counter.
- For persuasion-style rewrites, “longer” means “more persuasive framing,” not generic scene description.
- The builder is especially useful for prompt-based testing of video generators, even when the actual system under test is outside PyRIT.

## Known Limits In This Local Fork

- The builder currently expects existing file paths or URLs for media-heavy flows.
- It does not yet provide a polished browser upload flow for images or videos.
- The builder is for prompt creation and preview. It is not yet a complete end-to-end run console for every PyRIT scenario.
- Some converter previews depend on how well the helper model follows structured instructions. The current Grok fallback makes previews usable, but results can still vary.

## Useful Commands

List available backend initializers:

```bash
pyrit_backend --list-initializers
```

Run the backend without any initializer:

```bash
pyrit_backend --host 127.0.0.1 --port 8000
```

Build the frontend:

```bash
cd frontend
npm run build
```

Run frontend tests:

```bash
cd frontend
npm test
```

## Where To Look Next

Use the official Azure PyRIT docs for:

- notebooks and cookbooks
- memory backends
- scorers
- broader deployment patterns
- contribution guidance

Use this guide for:

- the local prompt-builder workflow in this fork
- Grok-backed prompt previews
- local port choices
- the simplest way to run this repo without spinning up unnecessary services
