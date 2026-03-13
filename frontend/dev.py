# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Cross-platform script to manage PyRIT UI development servers
"""

import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

# Ensure emoji and other Unicode characters don't crash on Windows consoles
# that use legacy encodings like cp1252. Characters that can't be encoded
# are replaced with '?' instead of raising UnicodeEncodeError.
sys.stdout.reconfigure(errors="replace")  # type: ignore[attr-defined]
sys.stderr.reconfigure(errors="replace")  # type: ignore[attr-defined]

# Determine workspace root (parent of frontend directory)
FRONTEND_DIR = Path(__file__).parent.absolute()
WORKSPACE_ROOT = FRONTEND_DIR.parent
DEFAULT_BACKEND_HOST = os.environ.get("PYRIT_BACKEND_HOST", "0.0.0.0")
DEFAULT_BACKEND_PORT = os.environ.get("PYRIT_BACKEND_PORT", "8000")
DEFAULT_FRONTEND_HOST = os.environ.get("PYRIT_FRONTEND_HOST", "127.0.0.1")
DEFAULT_FRONTEND_PORT = os.environ.get("PYRIT_FRONTEND_PORT", "3000")


def is_windows():
    return platform.system() == "Windows"


def sync_version():
    """Sync package.json version with PyRIT version"""
    try:
        # Get PyRIT version
        import pyrit

        pyrit_version = pyrit.__version__

        # Convert Python version format to npm format
        # e.g., "0.10.1.dev0" -> "0.10.1-dev.0"
        npm_version = pyrit_version.replace(".dev", "-dev.")

        # Read package.json
        package_json_path = FRONTEND_DIR / "package.json"
        with open(package_json_path) as f:
            package_data = json.load(f)

        # Update version if different
        if package_data.get("version") != npm_version:
            package_data["version"] = npm_version
            with open(package_json_path, "w") as f:
                json.dump(package_data, f, indent=2)
                f.write("\n")  # Add trailing newline
            print(f"📦 Updated frontend version to {npm_version}")
    except Exception as e:
        print(f"⚠️  Warning: Could not sync version: {e}")


def kill_process_by_pattern(pattern):
    """Kill processes matching a pattern (cross-platform)"""
    try:
        if is_windows():
            # Windows: use taskkill
            subprocess.run(
                f'taskkill /F /FI "COMMANDLINE like %{pattern}%" >nul 2>&1',
                shell=True,
                check=False,
            )
        else:
            # Unix: use pkill
            subprocess.run(["pkill", "-f", pattern], check=False, stderr=subprocess.DEVNULL)
    except Exception as e:
        print(f"Warning: Could not kill {pattern}: {e}")


def stop_servers():
    """Stop all running servers"""
    print("🛑 Stopping servers...")
    kill_process_by_pattern("pyrit.backend.main")
    kill_process_by_pattern("vite")
    time.sleep(1)
    print("✅ Servers stopped")


def start_backend(initializers: list[str] | None = None):
    """Start the FastAPI backend using pyrit_backend CLI.

    Args:
        initializers: Optional list of initializer names to run at startup.
            If not specified, no initializers are run.
    """
    print(f"🚀 Starting backend on port {DEFAULT_BACKEND_PORT}...")

    # Change to workspace root
    os.chdir(WORKSPACE_ROOT)

    # Set development mode environment variable
    env = os.environ.copy()
    env["PYRIT_DEV_MODE"] = "true"

    # Default to no initializers
    if initializers is None:
        initializers = []

    # Build command using pyrit_backend CLI
    cmd = [
        sys.executable,
        "-m",
        "pyrit.cli.pyrit_backend",
        "--host",
        DEFAULT_BACKEND_HOST,
        "--port",
        DEFAULT_BACKEND_PORT,
        "--log-level",
        "info",
    ]

    # Add initializers if specified
    if initializers:
        cmd.extend(["--initializers"] + initializers)

    # Start backend
    return subprocess.Popen(cmd, env=env)


def start_frontend():
    """Start the Vite frontend"""
    print(f"🎨 Starting frontend on port {DEFAULT_FRONTEND_PORT}...")

    # Change to frontend directory
    os.chdir(FRONTEND_DIR)

    # Start frontend process
    npm_cmd = "npm.cmd" if is_windows() else "npm"
    env = os.environ.copy()
    return subprocess.Popen(
        [
            npm_cmd,
            "run",
            "dev",
            "--",
            "--host",
            DEFAULT_FRONTEND_HOST,
            "--port",
            DEFAULT_FRONTEND_PORT,
        ],
        env=env,
    )


def start_servers():
    """Start both backend and frontend servers"""
    print("🚀 Starting PyRIT UI servers...")
    print()

    backend = start_backend()
    print("⏳ Waiting for backend to initialize...")
    time.sleep(5)  # Give backend more time to fully start up

    frontend = start_frontend()
    time.sleep(2)

    print()
    print("✅ Servers running!")
    print(f"   Backend:  http://localhost:{DEFAULT_BACKEND_PORT} (PID: {backend.pid})")
    print(f"   Frontend: http://localhost:{DEFAULT_FRONTEND_PORT} (PID: {frontend.pid})")
    print(f"   API Docs: http://localhost:{DEFAULT_BACKEND_PORT}/docs")
    print()
    print("Press Ctrl+C to stop")

    return backend, frontend


def wait_for_interrupt(backend, frontend):
    """Wait for user interrupt and cleanup"""
    try:
        # Wait for processes
        backend.wait()
        frontend.wait()
    except KeyboardInterrupt:
        print()
        print("🛑 Stopping servers...")

        # Terminate processes
        try:
            backend.terminate()
            frontend.terminate()

            # Wait for clean shutdown
            backend.wait(timeout=5)
            frontend.wait(timeout=5)
        except Exception:
            # Force kill if needed
            backend.kill()
            frontend.kill()

        print("✅ Servers stopped")


def main():
    """Main entry point"""
    # Sync version before any operation
    sync_version()

    if len(sys.argv) > 1:
        command = sys.argv[1].lower()

        if command == "stop":
            stop_servers()
            return
        if command == "restart":
            stop_servers()
            time.sleep(1)
        elif command == "start":
            pass  # Just start both
        elif command == "backend":
            print("🚀 Starting backend only...")
            backend = start_backend()
            print(f"✅ Backend running on http://localhost:{DEFAULT_BACKEND_PORT} (PID: {backend.pid})")
            print(f"   API Docs: http://localhost:{DEFAULT_BACKEND_PORT}/docs")
            print("\nPress Ctrl+C to stop")
            try:
                backend.wait()
            except KeyboardInterrupt:
                print("\n🛑 Stopping backend...")
                backend.terminate()
                backend.wait(timeout=5)
                print("✅ Backend stopped")
            return
        elif command == "frontend":
            print("🎨 Starting frontend only...")
            frontend = start_frontend()
            print(f"✅ Frontend running on http://localhost:{DEFAULT_FRONTEND_PORT} (PID: {frontend.pid})")
            print("\nPress Ctrl+C to stop")
            try:
                frontend.wait()
            except KeyboardInterrupt:
                print("\n🛑 Stopping frontend...")
                frontend.terminate()
                frontend.wait(timeout=5)
                print("✅ Frontend stopped")
            return
        else:
            print(f"Unknown command: {command}")
            print("Usage: python dev.py [start|stop|restart|backend|frontend]")
            sys.exit(1)

    # Start servers
    backend, frontend = start_servers()

    # Wait for interrupt
    wait_for_interrupt(backend, frontend)


if __name__ == "__main__":
    main()
