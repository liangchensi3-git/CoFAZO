"""Run the original optimizer oracle, storage and parity tests and the public CLI checks."""
import argparse
import os
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cuda", action="store_true", help="Require a CUDA device for the numerical tests")
    options = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="4", MKL_NUM_THREADS="4")
    for name in ("test_r34.py", "test_observer.py", "test_rc4.py", "test_39.py", "test_public.py"):
        command = [sys.executable, str(root / "tests" / name)]
        if options.cuda and name != "test_public.py":
            command.append("--require-cuda")
        subprocess.run(command, cwd=root, env=env, check=True)


if __name__ == "__main__":
    main()
