import subprocess
import sys
import os

def start_flask_app():
    print("Starting Flask server...")
    print(f"Python executable: {sys.executable}")
    print(f"Working directory: {os.getcwd()}")
    
    # Set environment variables
    env = os.environ.copy()
    env["FLASK_APP"] = "app.py"
    env["FLASK_DEBUG"] = "1"
    
    # Start Flask app
    process = subprocess.Popen(
        [sys.executable, "-m", "flask", "run", "--port=5000", "--debug"],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        universal_newlines=True
    )
    
    # Print output in real-time
    for line in process.stdout:
        print(line, end='')

if __name__ == "__main__":
    start_flask_app()
