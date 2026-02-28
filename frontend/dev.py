# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Cross-platform script to manage PyRIT UI development servers
"""

import json
import os
import platform
import signal
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
DEVPY_LOG_FILE = Path.home() / ".pyrit" / "dev.log"
DEVPY_PID_FILE = Path.home() / ".pyrit" / "dev.pid"


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


def find_pids_by_pattern(pattern):
    """Find PIDs of processes matching a pattern (cross-platform).

    Returns:
        list[int]: List of matching process IDs.
    """
    pids = []
    try:
        if is_windows():
            result = subprocess.run(
                ["wmic", "process", "where", f"CommandLine like '%{pattern}%'", "get", "ProcessId"],
                capture_output=True,
                text=True,
                check=False,
            )
            for line in result.stdout.strip().splitlines():
                line = line.strip()
                if line.isdigit():
                    pids.append(int(line))
        else:
            result = subprocess.run(
                ["pgrep", "-f", pattern],
                capture_output=True,
                text=True,
                check=False,
            )
            for line in result.stdout.strip().splitlines():
                line = line.strip()
                if line.isdigit():
                    pid = int(line)
                    # Don't include our own process
                    if pid != os.getpid():
                        pids.append(pid)
    except Exception:
        pass
    return pids


def kill_pids(pids):
    """Kill a list of processes by PID."""
    for pid in pids:
        try:
            os.kill(pid, signal.SIGTERM)
        except OSError:
            pass


def stop_servers():
    """Stop all running servers"""
    print("🛑 Stopping servers...")
    backend_pids = find_pids_by_pattern("pyrit.cli.pyrit_backend")
    frontend_pids = find_pids_by_pattern("node.*vite")
    # Also find any parent dev.py processes (detached wrappers)
    wrapper_pids = find_pids_by_pattern("frontend/dev.py")
    all_pids = backend_pids + frontend_pids + wrapper_pids
    if all_pids:
        print(f"   Killing PIDs: {all_pids}")
        kill_pids(all_pids)
        time.sleep(1)
    print("✅ Servers stopped")


def start_backend(*, config_file: str | None = None, initializers: list[str] | None = None):
    """Start the FastAPI backend using pyrit_backend CLI.

    Configuration (initializers, database, env files) is read automatically
    from ~/.pyrit/.pyrit_conf by the pyrit_backend CLI via ConfigurationLoader,
    unless overridden with *config_file*.
    """
    print("🚀 Starting backend on port 8000...")

    # Change to workspace root
    os.chdir(WORKSPACE_ROOT)

    # Set development mode environment variable
    env = os.environ.copy()
    env["PYRIT_DEV_MODE"] = "true"

    cmd = [
        sys.executable,
        "-m",
        "pyrit.cli.pyrit_backend",
        "--host",
        "0.0.0.0",
        "--port",
        "8000",
        "--log-level",
        "info",
    ]
    if config_file:
        cmd.extend(["--config-file", config_file])

    # Add initializers if specified
    if initializers:
        cmd.extend(["--initializers"] + initializers)

    # Start backend
    return subprocess.Popen(cmd, env=env)


def start_frontend():
    """Start the Vite frontend"""
    print("🎨 Starting frontend on port 3000...")

    # Change to frontend directory
    os.chdir(FRONTEND_DIR)

    # Start frontend process
    npm_cmd = "npm.cmd" if is_windows() else "npm"
    return subprocess.Popen([npm_cmd, "run", "dev"])


def start_servers(*, config_file: str | None = None):
    """Start both backend and frontend servers"""
    print("🚀 Starting PyRIT UI servers...")
    print()

    # Kill any stale processes from prior sessions
    stop_servers()

    backend = start_backend(config_file=config_file)
    print("⏳ Waiting for backend to initialize...")
    time.sleep(5)  # Give backend more time to fully start up

    frontend = start_frontend()
    time.sleep(2)

    print()
    print("✅ Servers running!")
    print(f"   Backend:  http://localhost:8000 (PID: {backend.pid})")
    print(f"   Frontend: http://localhost:3000 (PID: {frontend.pid})")
    print("   API Docs: http://localhost:8000/docs")
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


def start_detached(*, config_file: str | None = None):
    """Re-launch this script in a fully detached background process.

    The detached process writes stdout/stderr to DEVPY_LOG_FILE and its PID
    is recorded in DEVPY_PID_FILE so ``stop`` can find it.
    """
    DEVPY_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

    cmd = [sys.executable, str(Path(__file__).absolute())]
    if config_file:
        cmd.extend(["--config-file", config_file])

    log_fh = open(DEVPY_LOG_FILE, "w")
    proc = subprocess.Popen(
        cmd,
        stdout=log_fh,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    DEVPY_PID_FILE.write_text(str(proc.pid))
    print(f"🚀 dev.py started in background (PID: {proc.pid})")
    print(f"   Logs: {DEVPY_LOG_FILE}")
    print(f"   Stop: python {Path(__file__).name} stop")


def show_logs(*, follow: bool = False, lines: int = 50):
    """Show dev.py logs."""
    if not DEVPY_LOG_FILE.exists():
        print(f"No log file found at {DEVPY_LOG_FILE}")
        return
    if follow:
        subprocess.run(["tail", "-f", "-n", str(lines), str(DEVPY_LOG_FILE)])
    else:
        subprocess.run(["tail", "-n", str(lines), str(DEVPY_LOG_FILE)])


def main():
    """Main entry point"""
    # Sync version before any operation
    sync_version()

    # Extract --config-file and --detach from argv
    config_file: str | None = None
    detach = False
    argv = list(sys.argv[1:])
    if "--config-file" in argv:
        idx = argv.index("--config-file")
        if idx + 1 < len(argv):
            config_file = argv[idx + 1]
            argv = argv[:idx] + argv[idx + 2:]
        else:
            print("ERROR: --config-file requires a path argument")
            sys.exit(1)
    if "--detach" in argv:
        argv.remove("--detach")
        detach = True

    if argv:
        command = argv[0].lower()

        if command == "stop":
            stop_servers()
            return
        if command == "restart":
            stop_servers()
            time.sleep(1)
            # Fall through to start
        elif command == "start":
            pass  # Just start both
        elif command == "logs":
            follow = "-f" in argv or "--follow" in argv
            show_logs(follow=follow)
            return
        elif command == "backend":
            print("🚀 Starting backend only...")
            # Kill stale backend processes
            stale = find_pids_by_pattern("pyrit.cli.pyrit_backend")
            if stale:
                print(f"   Killing stale backend PIDs: {stale}")
                kill_pids(stale)
                time.sleep(1)
            backend = start_backend(config_file=config_file)
            print(f"✅ Backend running on http://localhost:8000 (PID: {backend.pid})")
            print("   API Docs: http://localhost:8000/docs")
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
            # Kill stale frontend processes
            stale = find_pids_by_pattern("node.*vite")
            if stale:
                print(f"   Killing stale frontend PIDs: {stale}")
                kill_pids(stale)
                time.sleep(1)
            frontend = start_frontend()
            print(f"✅ Frontend running on http://localhost:3000 (PID: {frontend.pid})")
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
            print(
                "Usage: python dev.py [start|stop|restart|backend|frontend|logs] "
                "[--config-file PATH] [--detach]"
            )
            sys.exit(1)

    # If --detach, re-launch in background and exit immediately
    if detach:
        start_detached(config_file=config_file)
        return

    # Start servers
    backend, frontend = start_servers(config_file=config_file)

    # Wait for interrupt
    wait_for_interrupt(backend, frontend)


if __name__ == "__main__":
    main()
