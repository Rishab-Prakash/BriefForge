
import streamlit as st
import subprocess
import sys

# Run the main BriefForge application
if __name__ == "__main__":
    subprocess.run([sys.executable, "-m", "streamlit", "run", "briefforge1.py", "--server.port=8501", "--server.address=0.0.0.0"])
