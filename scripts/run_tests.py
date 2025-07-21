#!/usr/bin/env python3
"""
Test runner script for plot2llm matplotlib tests.

This script provides various options for running the matplotlib test suite
with different configurations and reporting options.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path


def run_command(cmd, capture_output=False):
    """Run a command and optionally capture output."""
    print(f"Running: {cmd}")
    if capture_output:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        return result.returncode, result.stdout, result.stderr
    else:
        return subprocess.run(cmd, shell=True).returncode


def main():
    parser = argparse.ArgumentParser(description="Run plot2llm matplotlib tests")
    parser.add_argument(
        "--coverage", action="store_true", help="Run tests with coverage reporting"
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument(
        "--marker",
        "-m",
        type=str,
        help="Run tests with specific marker (e.g., 'unit', 'integration')",
    )
    parser.add_argument(
        "--fast", action="store_true", help="Run only fast tests (exclude slow marker)"
    )
    parser.add_argument("--file", type=str, help="Run specific test file")
    parser.add_argument(
        "--parallel",
        "-n",
        type=int,
        default=1,
        help="Run tests in parallel (number of workers)",
    )

    args = parser.parse_args()

    # Change to project root directory
    project_root = Path(__file__).parent.parent
    os.chdir(project_root)

    # Base pytest command
    cmd_parts = ["python", "-m", "pytest"]

    # Add coverage if requested
    if args.coverage:
        cmd_parts.extend(["--cov=plot2llm", "--cov-report=html", "--cov-report=term"])

    # Add verbosity
    if args.verbose:
        cmd_parts.append("-v")

    # Add marker selection
    if args.marker:
        cmd_parts.extend(["-m", args.marker])

    # Exclude slow tests if fast mode
    if args.fast:
        cmd_parts.extend(["-m", "not slow"])

    # Add parallel execution
    if args.parallel > 1:
        cmd_parts.extend(["-n", str(args.parallel)])

    # Add specific file if requested
    if args.file:
        cmd_parts.append(f"tests/{args.file}")
    else:
        # Run matplotlib tests by default
        cmd_parts.extend(
            ["tests/test_matplotlib_analyzer.py", "tests/test_matplotlib_formats.py"]
        )

    # Execute the command
    cmd = " ".join(cmd_parts)
    return_code = run_command(cmd)

    if return_code == 0:
        print("\n✅ All tests passed!")
        if args.coverage:
            print("📊 Coverage report generated in htmlcov/index.html")
    else:
        print("\n❌ Some tests failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
