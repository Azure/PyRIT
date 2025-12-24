# PyRIT Frontend

Modern TypeScript + React frontend for PyRIT, built with Fluent UI.

## Quick Start

### Installation

```bash
npm install
```

### Running the Frontend

```bash
# Start the frontend server
python dev.py start

# Stop the frontend server
python dev.py stop

# Restart the frontend server
python dev.py restart
```

The frontend will be available at `http://localhost:3000`

### Alternative: Using npm directly

```bash
npm run dev
```

### Building for Production

```bash
npm run build
```

The built files will be in the `dist/` directory.

## Tech Stack

- **React 18** - UI framework
- **TypeScript** - Type safety
- **Fluent UI v9** - Microsoft design system
- **Vite** - Fast build tool
Configure this in `vite.config.ts` if needed.
