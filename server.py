import subprocess
import threading
import os
import signal
import time
import sys
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Project root
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

# Backend and frontend paths
BACKEND_DIR = os.path.join(ROOT_DIR, 'backend')
FRONTEND_DIR = os.path.join(ROOT_DIR, 'frontend')
BACKEND_MAIN = os.path.join(BACKEND_DIR, 'app', 'main.py')

def run_backend():
    """Run the Flask backend"""
    logger.info("Starting Flask backend on http://localhost:5000...")
    if not os.path.exists(BACKEND_MAIN):
        logger.error(f"Backend main file not found: {BACKEND_MAIN}")
        return None
    try:
        backend_process = subprocess.Popen(
            [sys.executable, BACKEND_MAIN],
            cwd=BACKEND_DIR,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Stream backend output
        def stream_output():
            while True:
                stdout_line = backend_process.stdout.readline()
                stderr_line = backend_process.stderr.readline()
                if stdout_line:
                    logger.info(f"Backend: {stdout_line.strip()}")
                if stderr_line:
                    logger.error(f"Backend Error: {stderr_line.strip()}")
                if backend_process.poll() is not None:
                    break
        
        threading.Thread(target=stream_output, daemon=True).start()
        return backend_process
    except Exception as e:
        logger.error(f"Failed to start backend: {e}")
        return None

def run_frontend():
    """Run the frontend HTTP server"""
    logger.info("Starting frontend HTTP server on http://localhost:8000...")
    if not os.path.exists(FRONTEND_DIR):
        logger.error(f"Frontend directory not found: {FRONTEND_DIR}")
        return None
    try:
        frontend_process = subprocess.Popen(
            [sys.executable, "-m", "http.server", "8000"],
            cwd=FRONTEND_DIR,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Stream frontend output
        def stream_output():
            while True:
                stdout_line = frontend_process.stdout.readline()
                stderr_line = frontend_process.stderr.readline()
                if stdout_line:
                    logger.info(f"Frontend: {stdout_line.strip()}")
                if stderr_line:
                    logger.error(f"Frontend Error: {stderr_line.strip()}")
                if frontend_process.poll() is not None:
                    break
        
        threading.Thread(target=stream_output, daemon=True).start()
        return frontend_process
    except Exception as e:
        logger.error(f"Failed to start frontend: {e}")
        return None

def main():
    """Start both backend and frontend servers"""
    backend_process = None
    frontend_process = None
    
    try:
        # Start backend
        backend_process = run_backend()
        if backend_process is None:
            raise Exception("Backend failed to start")
        
        # Start frontend after a delay
        time.sleep(2)
        frontend_process = run_frontend()
        if frontend_process is None:
            raise Exception("Frontend failed to start")
        
        logger.info("Both servers started successfully!")
        logger.info("Backend: http://localhost:5000")
        logger.info("Frontend: http://localhost:8000")
        logger.info("Press Ctrl+C to stop...")
        
        # Keep main thread alive
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        logger.info("Shutting down servers...")
        for process in [backend_process, frontend_process]:
            if process:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
        logger.info("Servers stopped.")
    except Exception as e:
        logger.error(f"Error: {e}")
        for process in [backend_process, frontend_process]:
            if process:
                process.terminate()

if __name__ == '__main__':
    main()
