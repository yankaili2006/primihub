#!/usr/bin/env python3
"""
PrimiHub FL Test Runner

Usage:
    python run_fl_tests.py                    # Run all FL tests
    python run_fl_tests.py -v                 # Verbose
    python run_fl_tests.py -k "model"         # Filter by keyword
    python run_fl_tests.py --coverage         # Show coverage
"""
import sys
import subprocess
import os

FL_TEST_DIR = os.path.join(os.path.dirname(__file__),
                           "python", "primihub", "FL", "tests")


def main():
    args = sys.argv[1:] if len(sys.argv) > 1 else []
    cmd = [sys.executable, "-m", "pytest", FL_TEST_DIR, "-v"] + args
    if "--coverage" in args:
        args.remove("--coverage")
        cmd = [
            sys.executable, "-m", "pytest", FL_TEST_DIR, "-v",
            "--cov=primihub.FL",
            "--cov-report=term-missing",
            "--cov-report=html:fl_coverage_report",
        ] + args
        print("Coverage report will be in fl_coverage_report/index.html")

    print(f"Running: {' '.join(cmd)}")
    sys.exit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
