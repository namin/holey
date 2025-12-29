#!/usr/bin/env python3
"""Regenerate example output files (.txt and .smt) from .py SAT files."""

import os
import subprocess
import sys
from pathlib import Path


def extract_smt(txt_content: str) -> str:
    """Extract SMT content from puzzle_solver output."""
    lines = txt_content.split('\n')
    smt_lines = []
    in_smt = False

    for line in lines:
        if line.strip() == '### smt2':
            in_smt = True
            continue
        if in_smt:
            smt_lines.append(line)
            if line.strip() == '(get-model)':
                break

    return '\n'.join(smt_lines)


def run_solver(py_file: Path, no_bounded: bool = False) -> str:
    """Run puzzle_solver.py on a SAT file and return output."""
    cmd = [sys.executable, 'puzzle_solver.py', '--sat-file', str(py_file)]
    if no_bounded:
        cmd.append('--no-bounded-list')

    env = os.environ.copy()
    env['TRUNCATE'] = 'true'

    result = subprocess.run(cmd, capture_output=True, text=True, env=env)
    return result.stdout + result.stderr


def regenerate_example(py_file: Path, examples_dir: Path):
    """Regenerate .txt and .smt files for a single example."""
    base_name = py_file.stem

    # Generate standard (bounded) output
    print(f"Processing {py_file.name}...")
    output = run_solver(py_file)

    txt_file = examples_dir / f"{base_name}.txt"
    smt_file = examples_dir / f"{base_name}.smt"

    txt_file.write_text(output)
    print(f"  Wrote {txt_file.name}")

    smt_content = extract_smt(output)
    if smt_content.strip():
        smt_file.write_text(smt_content)
        print(f"  Wrote {smt_file.name}")

    # Check if no_bounded version exists
    no_bounded_smt = examples_dir / f"{base_name}_no_bounded.smt"
    no_bounded_txt = examples_dir / f"{base_name}_no_bounded.txt"

    if no_bounded_smt.exists() or no_bounded_txt.exists():
        print(f"  Regenerating no_bounded version...")
        output = run_solver(py_file, no_bounded=True)

        no_bounded_txt.write_text(output)
        print(f"  Wrote {no_bounded_txt.name}")

        smt_content = extract_smt(output)
        if smt_content.strip():
            no_bounded_smt.write_text(smt_content)
            print(f"  Wrote {no_bounded_smt.name}")


def main():
    examples_dir = Path('examples')

    if not examples_dir.exists():
        print("Error: examples/ directory not found")
        sys.exit(1)

    # Find all .py files in examples/
    py_files = sorted(examples_dir.glob('*.py'))

    if not py_files:
        print("No .py files found in examples/")
        sys.exit(1)

    print(f"Found {len(py_files)} example files\n")

    for py_file in py_files:
        regenerate_example(py_file, examples_dir)
        print()

    print("Done!")


if __name__ == '__main__':
    main()
