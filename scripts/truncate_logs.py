#!/usr/bin/env python3
"""
Log truncation utility for PrimiHub.
This script can be used to truncate existing log files to prevent them from growing too large.
"""

import os
import sys
import argparse
from pathlib import Path


def truncate_log_file(file_path, max_lines=5000, keep_first=1000, keep_last=4000):
    """
    Truncate a log file by keeping the first and last portions.

    Args:
        file_path: Path to the log file
        max_lines: Maximum total lines to keep
        keep_first: Number of lines to keep from the beginning
        keep_last: Number of lines to keep from the end
    """
    if not os.path.exists(file_path):
        print(f"File {file_path} does not exist")
        return False

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        total_lines = len(lines)
        if total_lines <= max_lines:
            print(
                f"File {file_path} has {total_lines} lines (<= {max_lines}), no truncation needed"
            )
            return True

        print(
            f"Truncating {file_path}: {total_lines} -> {keep_first + keep_last} lines"
        )

        # Keep first and last portions
        truncated_lines = (
            lines[:keep_first]
            + [
                f"\n[LOG TRUNCATED - {total_lines - keep_first - keep_last} LINES REMOVED]\n\n"
            ]
            + lines[-keep_last:]
        )

        # Write truncated content back
        with open(file_path, "w", encoding="utf-8") as f:
            f.writelines(truncated_lines)

        print(f"Successfully truncated {file_path}")
        return True

    except Exception as e:
        print(f"Error truncating {file_path}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Truncate PrimiHub log files")
    parser.add_argument(
        "--log-dir", default="./log", help="Directory containing log files"
    )
    parser.add_argument(
        "--max-lines", type=int, default=5000, help="Maximum lines per log file"
    )
    parser.add_argument(
        "--keep-first", type=int, default=1000, help="Lines to keep from beginning"
    )
    parser.add_argument(
        "--keep-last", type=int, default=4000, help="Lines to keep from end"
    )
    parser.add_argument("--pattern", default="*.log", help="Log file pattern to match")

    args = parser.parse_args()

    log_dir = Path(args.log_dir)
    if not log_dir.exists():
        print(f"Log directory {log_dir} does not exist")
        return 1

    log_files = list(log_dir.glob(args.pattern))
    if not log_files:
        print(f"No log files found matching pattern {args.pattern} in {log_dir}")
        return 1

    print(f"Found {len(log_files)} log files to process")

    success_count = 0
    for log_file in log_files:
        if truncate_log_file(log_file, args.max_lines, args.keep_first, args.keep_last):
            success_count += 1

    print(f"Successfully processed {success_count}/{len(log_files)} log files")
    return 0


if __name__ == "__main__":
    sys.exit(main())
