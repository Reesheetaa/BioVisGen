"""Single entry point for the BioVisGen interactive system."""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SYSTEM = ROOT / "biovisgen_agentic_architecture"
sys.path.insert(0, str(SYSTEM))
os.chdir(SYSTEM)

from main import demo

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
