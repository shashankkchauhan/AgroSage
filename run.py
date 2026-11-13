"""
AgroSage Startup Script
Runs both FastAPI backend and Streamlit frontend in background processes.
"""
import subprocess
import sys
import os
import time

ROOT = os.path.dirname(os.path.abspath(__file__))

def start():
    print("=" * 60)
    print("  🌾 AgroSage — Smart Farming AI Assistant")
    print("  Powered by IBM Granite & IBM watsonx.ai")
    print("=" * 60)

    print("\n[1/2] Starting FastAPI backend on http://localhost:8000 ...")
    backend = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "backend.main:app",
         "--host", "0.0.0.0", "--port", "8000", "--reload"],
        cwd=ROOT,
    )
    time.sleep(2)

    print("[2/2] Starting Streamlit frontend on http://localhost:8501 ...")
    frontend = subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run", "frontend/app.py",
         "--server.port", "8501", "--server.headless", "true"],
        cwd=ROOT,
    )

    print("\n✅ AgroSage is running!")
    print("   Frontend : http://localhost:8501")
    print("   API Docs : http://localhost:8000/docs")
    print("   Press Ctrl+C to stop.\n")

    try:
        backend.wait()
    except KeyboardInterrupt:
        print("\nShutting down...")
        backend.terminate()
        frontend.terminate()

if __name__ == "__main__":
    start()
