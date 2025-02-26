import subprocess
import os

# Ensure the script runs in the correct directory
base_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(base_dir)

# Run FastAPI with Uvicorn, ensuring the module is found
subprocess.run(["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"])
