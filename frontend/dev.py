# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Minimal script to manage PyRIT frontend (no backend)
"""

import platform
import subprocess
import sys
import time
from pathlib import Path

# Determine workspace root (parent of frontend directory)
FRONTEND_DIR = Path(__file__).parent.absolute()


def is_windows():
    return platform.system() == "Windows"


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


def stop_frontend():
    """Stop the frontend server"""
    print("🛑 Stopping frontend...")
    kill_process_by_pattern("vite")
    time.sleep(1)
    print("✅ Frontend stopped")


def start_frontend():
    """Start the Vite frontend"""
    print("🎨 Starting minimal PyRIT frontend on port 3000...")
    print("    (No backend - this is a standalone demo)")

    # Start frontend process
    npm_cmd = "npm.cmd" if is_windows() else "npm"

    if is_windows():
        frontend = subprocess.Popen(
            [npm_cmd, "run", "dev"],
            cwd=FRONTEND_DIR,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
        )
    else:
        frontend = subprocess.Popen([npm_cmd, "run", "dev"], cwd=FRONTEND_DIR)

    time.sleep(2)
    print()
    print("✅ Frontend running!")
    print(f"   URL: http://localhost:3000 (PID: {frontend.pid})")
    print()
    print("Press Ctrl+C to stop")

    return frontend


def wait_for_interrupt(frontend):
    """Wait for user interrupt and cleanup"""
    try:
        frontend.wait()
    except KeyboardInterrupt:
        print()
        print("🛑 Stopping frontend...")

        try:
            if not is_windows():
                frontend.terminate()
            frontend.wait(timeout=5)
        except:  # noqa: E722
            frontend.kill()

        print("✅ Frontend stopped")


def main():
    """Main entry point"""
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()

        if command == "stop":
            stop_frontend()
            return
        elif command == "restart":
            stop_frontend()
            time.sleep(1)
        elif command in ["start", "frontend"]:
            pass  # Start frontend
        else:
            print(f"Unknown command: {command}")
            print("Usage: python dev.py [start|stop|restart]")
            sys.exit(1)

    # Start frontend
    frontend = start_frontend()

    # Wait for interrupt
    wait_for_interrupt(frontend)


if __name__ == "__main__":
    main()
