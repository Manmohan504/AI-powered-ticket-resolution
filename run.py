import subprocess
import sys
import os

if __name__ == "__main__":
    app_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "app", "app.py")
    env = os.environ.copy()
    env["PYTHONPATH"] = os.path.join(os.path.dirname(os.path.abspath(__file__)), "app")
    
    print(f"Starting AI Ticket Resolution System...")
    print(f"App: {app_path}")
    
    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", app_path, "--server.headless", "true"],
        env=env,
    )
