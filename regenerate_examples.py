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


def run_solver(py_file: Path, flag: str = "") -> str:
    """Run puzzle_solver.py on a SAT file and return output."""
    cmd = [sys.executable, 'puzzle_solver.py', '--sat-file', str(py_file)]
    cmd += ['--smtlib-backends', 'z3', 'cvc5']
    if flag:
        cmd.append(flag)

    env = os.environ.copy()
    env['TRUNCATE'] = 'false'

    result = subprocess.run(cmd, capture_output=True, text=True, env=env)
    return result.stdout + result.stderr


def regenerate_example(py_file: Path, examples_dir: Path):
    """Regenerate .txt and .smt files for a single example."""
    base_name = py_file.stem

    # Generate standard (bounded) output
    print(f"Processing {py_file.name}...")

    for (fn, flag) in [('', ''), ('_no_bounded', '--no-bounded-list'), ('_no_ite', '--no-ite')]:
        txt_file = examples_dir / f"{base_name}{fn}.txt"
        smt_file = examples_dir / f"{base_name}{fn}.smt"

        if txt_file.exists() or smt_file.exists():
            print(f"  Regenerating{'' if not fn else ' '}{fn} version...")
            output = run_solver(py_file, flag=flag)
            txt_file.write_text(output)
            print(f"  Wrote {txt_file.name}")

            smt_content = extract_smt(output)
            if smt_content.strip():
                smt_file.write_text(smt_content)
                print(f"  Wrote {smt_file.name}")

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
