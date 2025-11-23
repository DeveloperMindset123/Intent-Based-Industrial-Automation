#!/usr/bin/env python3
"""
Test runner that uses the .venv Python interpreter and captures all output.
"""
import sys
import subprocess
from pathlib import Path

# Get the script directory
script_dir = Path(__file__).parent
venv_python = script_dir / ".venv" / "bin" / "python"

# Path to the run script
run_script = script_dir / "multi_agent_implementation_demo" / "run_with_output_capture.py"

if __name__ == "__main__":
    print(f"Using Python: {venv_python}")
    print(f"Running script: {run_script}")
    print("="*70)
    
    # Run the script with output capture
    try:
        result = subprocess.run(
            [str(venv_python), str(run_script)],
            cwd=str(script_dir),
            capture_output=False,  # Show output in real-time
            text=True,
            check=False
        )
        sys.exit(result.returncode)
    except Exception as e:
        print(f"Error running script: {e}")
        sys.exit(1)

