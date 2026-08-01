"""
Launch the AI Review System — backend server + open browser.

Usage:
    python start_system.py
    python start_system.py --port 8000 --no-browser
"""

import argparse
import subprocess
import sys
import time
import webbrowser
from pathlib import Path

PROJECT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description="Launch AI Review System")
    parser.add_argument("--port", type=int, default=8000, help="Server port (default: 8000)")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Bind address (default: 127.0.0.1)")
    parser.add_argument("--no-browser", action="store_true", help="Don't open browser automatically")
    parser.add_argument("--venv", type=str, default="venv", help="Virtual env directory name (default: venv)")
    args = parser.parse_args()

    url = f"http://{args.host}:{args.port}"

    # Detect Python interpreter — prefer venv if it exists
    venv_python = PROJECT / args.venv / "Scripts" / "python.exe"
    python_exe = str(venv_python) if venv_python.exists() else sys.executable

    if venv_python.exists():
        print(f"[OK] Using venv: {venv_python}")
    else:
        print(f"[!] venv not found at {venv_python}, using system Python: {python_exe}")

    print("=" * 60)
    print("  AI Review System — 论文审稿与攻击防御")
    print("=" * 60)
    print()
    print(f"  Backend:  {url}")
    print(f"  API Docs: {url}/docs")
    print(f"  Health:   {url}/api/health")
    print()

    if not args.no_browser:
        print("  Browser will open automatically...")
        # Delay browser open to let server start
        webbrowser.open(url)

    print()
    print("  Press Ctrl+C to stop the server.")
    print("=" * 60)
    print()

    server_script = str(PROJECT / "review_app.py")

    # Use uvicorn directly for better control
    try:
        subprocess.run(
            [python_exe, "-m", "uvicorn", "review_app:app",
             "--host", args.host,
             "--port", str(args.port),
             "--log-level", "info"],
            cwd=str(PROJECT),
            check=True,
        )
    except KeyboardInterrupt:
        print("\nServer stopped.")
    except subprocess.CalledProcessError:
        # Fallback: run review_app.py directly
        print("[!] uvicorn module failed, trying direct launch...")
        subprocess.run([python_exe, server_script], cwd=str(PROJECT))


if __name__ == "__main__":
    main()
