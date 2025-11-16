# DevContainer Updates for PyRIT UI

## Changes Made

### 1. **Dockerfile** ✅
- Added Node.js 20.x installation
- Added npm (latest version)
- Dependencies will be available for frontend development

### 2. **devcontainer.json** ✅
- Added port forwarding for:
  - `3000` - Frontend (Vite dev server)
  - `8000` - Backend (FastAPI)
  - Kept existing: `4213`, `5000`, `8888`
- Added VS Code extensions:
  - `dbaeumer.vscode-eslint` - ESLint for TypeScript/JavaScript
  - `esbenp.prettier-vscode` - Code formatting

### 3. **devcontainer_setup.sh** ✅
- Added automatic npm install for frontend dependencies
- Runs after Python dependencies are installed

## Next Steps

### Rebuild the DevContainer

**Option 1: Using VS Code Command Palette**
1. Press `Ctrl+Shift+P` (or `Cmd+Shift+P` on Mac)
2. Type: "Dev Containers: Rebuild Container"
3. Select it and wait for rebuild

**Option 2: Using Command Palette (without cache)**
1. Press `Ctrl+Shift+P` (or `Cmd+Shift+P` on Mac)
2. Type: "Dev Containers: Rebuild Container Without Cache"
3. This ensures a clean build with all new packages

### After Rebuild

The container will automatically:
1. ✅ Install Node.js and npm
2. ✅ Install Python dependencies (including FastAPI)
3. ✅ Install frontend npm packages
4. ✅ Forward ports 3000 and 8000

### Start the Application

Once rebuilt, you can:

```bash
# Start both servers
./start-dev.sh

# Or start separately
# Terminal 1:
python -m pyrit.backend.main

# Terminal 2:
cd frontend && npm run dev
```

### Access URLs (after starting)
- Frontend: http://localhost:3000
- Backend API: http://localhost:8000
- API Docs: http://localhost:8000/docs

## What's Included in the Container

- Python 3.11 (conda environment)
- Node.js 20.x
- npm (latest)
- All PyRIT dependencies
- FastAPI & Uvicorn
- All frontend npm packages (React, TypeScript, Fluent UI, etc.)

## Verification

After rebuild, verify the setup:
```bash
./verify-setup.sh
```

This will check:
- ✓ Backend files exist
- ✓ Frontend files exist
- ✓ FastAPI installed
- ✓ Uvicorn installed
- ✓ Node.js/npm available
- ✓ node_modules installed
