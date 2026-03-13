# PyRIT Frontend

Modern TypeScript + React frontend for PyRIT, built with Fluent UI.

This fork ships a prompt-builder UI for red teaming video generation systems. The builder keeps the full converter list, then adds a small attack-starter layer on top:

- Curated starter families for Crescendo, TAP, RedTeaming, and single-turn probes
- Character-driven preset fields that fill the normal prompt box without locking it
- An optional blocked-word rewrite step before the selected converter runs
- Multiple text versions when the chosen converter supports that flow
- Optional helper-generated reference images for image-based attack setups

## Development

```bash
# Install dependencies
npm install

# Start both backend and frontend (cross-platform)
python dev.py start
# OR use npm script
npm run start

# Start backend only (with airt initializer by default)
python dev.py backend

# Start frontend only (backend must be started separately)
python dev.py frontend
# OR
npm run dev

# Restart both servers
python dev.py restart
# OR
npm run restart

# Stop all servers
python dev.py stop
# OR
npm run stop

# Build for production
npm run build

# Preview production build
npm run preview
```

### Backend CLI

The backend uses `pyrit_backend` CLI which supports initializers:

```bash
# Start with default airt initializer (loads targets from env vars)
pyrit_backend --initializers airt

# Start without initializers
pyrit_backend

# Start with custom initialization script
pyrit_backend --initialization-scripts ./my_targets.py

# List available initializers
pyrit_backend --list-initializers

# Custom host/port
pyrit_backend --host 127.0.0.1 --port 8080
```

### Recommended Local Builder Setup

If you want the builder to behave like this fork expects, use the Grok-backed initializer from the repo root:

```bash
PYRIT_CORS_ORIGINS="http://127.0.0.1:4174,http://localhost:4174,http://127.0.0.1:3000,http://localhost:3000" \
pyrit_backend --host 127.0.0.1 --port 8002 --initialization-scripts scripts/grok_builder_initializer.py
```

Then start the frontend:

```bash
cd frontend
VITE_API_URL="http://127.0.0.1:8002/api" npm run dev -- --host 127.0.0.1 --port 4174
```

### Helper Model Environment

Minimum helper-model setup:

```env
OPENAI_CHAT_ENDPOINT="https://api.x.ai/v1"
OPENAI_CHAT_KEY="YOUR_GROK_KEY"
OPENAI_CHAT_MODEL="grok-4-latest"
OPENAI_CHAT_UNDERLYING_MODEL="grok-4-latest"
```

Optional reference-image support:

```env
OPENAI_IMAGE_MODEL="gpt-image-1"
```

Optional image overrides:

```env
OPENAI_IMAGE_ENDPOINT="https://api.x.ai/v1"
OPENAI_IMAGE_KEY="YOUR_GROK_KEY"
OPENAI_IMAGE_UNDERLYING_MODEL="gpt-image-1"
```

If you leave `OPENAI_IMAGE_MODEL` unset, the rest of the builder still works and the reference-image controls stay disabled.

**Development Mode**: The `dev.py` script sets `PYRIT_DEV_MODE=true` so the backend expects the frontend to run separately on port 3000.

**Production Mode**: When installed from PyPI, the backend serves the bundled frontend and will exit if frontend files are missing.

## Stack

- **React 18** - UI framework
- **TypeScript** - Type safety
- **Fluent UI v9** - Microsoft design system
- **Vite** - Fast build tool
- **Axios** - HTTP client

## Testing

```bash
# Unit & Integration Tests (Jest + React Testing Library)
npm test              # Run all tests
npm run test:watch    # Watch mode for development
npm run test:coverage # Run with coverage report (85%+ threshold)

# End-to-End Tests (Playwright)
npm run test:e2e          # Run headless (auto-starts frontend + backend via dev.py)
npm run test:e2e:headed   # Run with visible browser windows (requires display)
npm run test:e2e:ui       # Interactive UI mode (requires display)
```

E2E tests use `dev.py` to automatically start both frontend and backend servers. If servers are already running, they will be reused.

> **Note**: `test:e2e:ui` and `test:e2e:headed` require a graphical display and won't work in headless environments like devcontainers. Use `npm run test:e2e` for CI/headless testing.

## Configuration

The frontend proxies API requests to `http://localhost:8000` in development.
Configure this in `vite.config.ts` if needed.

Builder-specific backend endpoints:

- `GET /api/builder/config`
- `POST /api/builder/build`
- `POST /api/builder/reference-image`
