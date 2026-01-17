#!/usr/bin/env python3
"""Launch script for Recipier frontend."""
import subprocess
import sys
from pathlib import Path


def main():
    """Launch the Streamlit frontend."""
    # Get the path to app.py
    project_root = Path(__file__).parent
    app_path = project_root / 'frontend' / 'app.py'

    if not app_path.exists():
        print(f"Error: {app_path} not found")
        sys.exit(1)

    # Launch Streamlit
    cmd = [
        sys.executable, '-m', 'streamlit', 'run',
        str(app_path),
        '--server.headless', 'true',
        '--browser.gatherUsageStats', 'false',
    ]

    print(f"Starting Recipier frontend...")
    print(f"Running: {' '.join(cmd)}")

    try:
        subprocess.run(cmd, cwd=str(project_root))
    except KeyboardInterrupt:
        print("\nShutting down...")
        sys.exit(0)


if __name__ == '__main__':
    main()
