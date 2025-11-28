import os
import sys
import subprocess
from pathlib import Path

REQ_FILE = Path(__file__).parent / "requirements.txt"


def run(cmd):
    print(f"[install] Running: {' '.join(cmd)}")
    try:
        subprocess.check_call(cmd)
    except subprocess.CalledProcessError as e:
        print(f"[install] Command failed with code {e.returncode}: {' '.join(cmd)}")
        sys.exit(e.returncode)


def main():
    if not REQ_FILE.exists():
        print(f"[install] requirements.txt not found at: {REQ_FILE}")
        sys.exit(1)

    # Ensure pip is available and up to date for the current interpreter.
    run([sys.executable, "-m", "pip", "install", "--upgrade", "pip"]) 

    # Install dependencies from requirements.txt
    run([sys.executable, "-m", "pip", "install", "-r", str(REQ_FILE)])

    print("[install] All requirements have been installed successfully.")


if __name__ == "__main__":
    main()
