#!/usr/bin/env python3
"""
Find regressions between two benchmark log files.
A regression is a puzzle that was solved in LOG1 but not in LOG2.
"""

import sys
import re


def parse_log(filename):
    """Parse a log file and return a set of solved puzzle names."""
    solved = set()

    with open(filename, 'r') as f:
        for line in f:
            # Match "Yes! Solved for puzzle PUZZLE_NAME"
            match = re.match(r'Yes! Solved for puzzle\s+(.+)', line)
            if match:
                solved.add(match.group(1).strip())

    return solved


def main():
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} LOG1 LOG2")
        print("Finds puzzles solved in LOG1 but not in LOG2 (regressions)")
        sys.exit(1)

    log1, log2 = sys.argv[1], sys.argv[2]

    solved1 = parse_log(log1)
    solved2 = parse_log(log2)

    regressions = sorted(solved1 - solved2)
    improvements = sorted(solved2 - solved1)

    print(f"LOG1 ({log1}): {len(solved1)} puzzles solved")
    print(f"LOG2 ({log2}): {len(solved2)} puzzles solved")
    print()

    print(f"Regressions ({len(regressions)} puzzles solved in LOG1 but not LOG2):")
    print(" ".join(regressions))

    print()
    print(f"Improvements ({len(improvements)} puzzles solved in LOG2 but not LOG1):")
    print(" ".join(improvements))


if __name__ == "__main__":
    main()
