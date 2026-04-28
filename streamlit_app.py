import subprocess
import sys

# Run the actual app from app/main.py
subprocess.run([sys.executable, "-m", "streamlit", "run", "app/main.py"])