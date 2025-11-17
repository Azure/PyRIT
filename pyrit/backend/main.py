# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
FastAPI application entry point for PyRIT backend
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from pyrit.setup.initialization import initialize_pyrit
from pyrit.backend.routes import health, chat, targets, config, converters, convert

app = FastAPI(
    title="PyRIT API",
    description="Python Risk Identification Tool for LLMs - REST API",
    version="0.10.0",
)

# Initialize PyRIT on startup to load .env and .env.local files
@app.on_event("startup")
async def startup_event():
    # Use in-memory to avoid database initialization delays
    initialize_pyrit(memory_db_type="InMemory")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],  # Vite default ports
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Include routers
app.include_router(health.router, prefix="/api", tags=["health"])
app.include_router(chat.router, prefix="/api", tags=["chat"])
app.include_router(targets.router, prefix="/api", tags=["targets"])
app.include_router(config.router, prefix="/api/config", tags=["config"])
app.include_router(converters.router, tags=["converters"])
app.include_router(convert.router, tags=["convert"])


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "error": str(exc)},
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
