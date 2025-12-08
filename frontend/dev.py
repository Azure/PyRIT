#!/usr/bin/env python3
"""
Cross-platform script to manage PyRIT UI development servers
"""

import os
import sys
import signal
import subprocess
import time
import platform
from pathlib import Path

# Determine workspace root (parent of frontend directory)
FRONTEND_DIR = Path(__file__).parent.absolute()
WORKSPACE_ROOT = FRONTEND_DIR.parent


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


def stop_servers():
    """Stop all running servers"""
    print("🛑 Stopping servers...")
    kill_process_by_pattern("pyrit.backend.main")
    kill_process_by_pattern("vite")
    time.sleep(1)
    print("✅ Servers stopped")


def start_backend():
    """Start the FastAPI backend"""
    print("🚀 Starting backend on port 8000...")
    
    # Change to workspace root
    os.chdir(WORKSPACE_ROOT)
    
    # Set development mode environment variable
    env = os.environ.copy()
    env["PYRIT_DEV_MODE"] = "true"
    
    # Start backend with uvicorn
    if is_windows():
        backend = subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "pyrit.backend.main:app", 
             "--host", "0.0.0.0", "--port", "8000", "--log-level", "info"],
            env=env,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if is_windows() else 0,
        )
    else:
        backend = subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "pyrit.backend.main:app", 
             "--host", "0.0.0.0", "--port", "8000", "--log-level", "info"],
            env=env,
        )
    
    return backend


def start_frontend():
    """Start the Vite frontend"""
    print("🎨 Starting frontend on port 3000...")
    
    # Change to frontend directory
    os.chdir(FRONTEND_DIR)
    
    # Start frontend process
    npm_cmd = "npm.cmd" if is_windows() else "npm"
    
    if is_windows():
        frontend = subprocess.Popen(
            [npm_cmd, "run", "dev"],
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if is_windows() else 0,
        )
    else:
        frontend = subprocess.Popen([npm_cmd, "run", "dev"])
    
    return frontend


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
            if is_windows():
                backend.send_signal(signal.CTRL_BREAK_EVENT)
                frontend.send_signal(signal.CTRL_BREAK_EVENT)
            else:
                backend.terminate()
                frontend.terminate()
            
            # Wait for clean shutdown
            backend.wait(timeout=5)
            frontend.wait(timeout=5)
        except:
            # Force kill if needed
            backend.kill()
            frontend.kill()
        
        print("✅ Servers stopped")


def main():
    """Main entry point"""
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == "stop":
            stop_servers()
            return
        elif command == "restart":
            stop_servers()
            time.sleep(1)
        elif command == "start":
            pass  # Just start both
        elif command == "backend":
            print("🚀 Starting backend only...")
            backend = start_backend()
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
            print("Usage: python dev.py [start|stop|restart|backend|frontend]")
            sys.exit(1)
    
    # Start servers
    backend, frontend = start_servers()
    
    # Wait for interrupt
    wait_for_interrupt(backend, frontend)


if __name__ == "__main__":
    main()
