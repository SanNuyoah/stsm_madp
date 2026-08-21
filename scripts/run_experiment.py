#!/usr/bin/env python3
import os
import subprocess
import sys


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    shell_entry = os.path.join(script_dir, "run_experiments.sh")
    return subprocess.call(["bash", shell_entry] + sys.argv[1:])


if __name__ == "__main__":
    sys.exit(main())
