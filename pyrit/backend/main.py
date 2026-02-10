# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
FastAPI application entry point for PyRIT backend.
"""

import os
import sys
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

import pyrit
from pyrit.backend.middleware import register_error_handlers
from pyrit.backend.routes import attacks, converters, health, labels, targets, version
from pyrit.setup.initialization import initialize_pyrit_async

# Check for development mode from environment variable
DEV_MODE = os.getenv("PYRIT_DEV_MODE", "false").lower() == "true"


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Manage application startup and shutdown lifecycle."""
    # Startup: initialize PyRIT to load .env and .env.local files
    await initialize_pyrit_async(memory_db_type="SQLite")
    yield
    # Shutdown: nothing to clean up currently


app = FastAPI(
    title="PyRIT API",
    description="Python Risk Identification Tool for LLMs - REST API",
    version=pyrit.__version__,
    lifespan=lifespan,
)

# Register RFC 7807 error handlers
register_error_handlers(app)


# Configure CORS
_default_origins = "http://localhost:3000,http://localhost:5173"
_cors_origins = [o.strip() for o in os.getenv("PYRIT_CORS_ORIGINS", _default_origins).split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Include API routes
app.include_router(attacks.router, prefix="/api", tags=["attacks"])
app.include_router(targets.router, prefix="/api", tags=["targets"])
app.include_router(converters.router, prefix="/api", tags=["converters"])
app.include_router(labels.router, prefix="/api", tags=["labels"])
app.include_router(health.router, prefix="/api", tags=["health"])
app.include_router(version.router, tags=["version"])


def setup_frontend() -> None:
    """Set up frontend static file serving (only called when running as main script)."""
    frontend_path = Path(__file__).parent / "frontend"

    if DEV_MODE:
        # Development mode: frontend served separately by Vite
        print("🔧 Running in DEVELOPMENT mode - frontend should be running on port 3000")
    elif frontend_path.exists():
        # Production mode: serve bundled frontend
        print(f"✅ Serving frontend from {frontend_path}")
        app.mount("/", StaticFiles(directory=str(frontend_path), html=True), name="frontend")
    else:
        # Production mode but no frontend found - this is an error
        print("❌ ERROR: Frontend not found!")
        print(f"   Expected location: {frontend_path}")
        print("   The frontend must be built and included in the package.")
        print("   Run: python build_scripts/prepare_package.py")
        sys.exit(1)


if __name__ == "__main__":
    import uvicorn

    setup_frontend()
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
