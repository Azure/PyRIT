# PyRIT Frontend

Modern TypeScript + React frontend for PyRIT, built with Fluent UI.

## Development

```bash
# Install dependencies
npm install

# Start both backend and frontend (cross-platform)
npm run start

# Restart both servers
npm run restart

# Stop all servers
npm run stop

# Or run Vite dev server only
npm run dev

# Build for production
npm run build

# Preview production build
npm run preview
```

## Stack

- **React 18** - UI framework
- **TypeScript** - Type safety
- **Fluent UI v9** - Microsoft design system
- **Vite** - Fast build tool
- **Axios** - HTTP client

## Project Structure

```
src/
├── components/     # React components
│   ├── Chat/      # Chat interface
│   ├── Layout/    # Layout components
│   └── Sidebar/   # Navigation sidebar
├── services/      # API client
├── types/         # TypeScript types
└── styles/        # Global styles
```

## Configuration

The frontend proxies API requests to `http://localhost:8000` in development.
Configure this in `vite.config.ts` if needed.
