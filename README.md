# Holey

A Python library for program synthesis and symbolic execution that combines Z3's constraint solving with LLM-guided synthesis. Put holes in your Python code and let `holey` fill them using formal constraints, natural language specifications, or both.

The symbolic execution is
inspired by Philip Zucker's blog post [_"Symbolic Execution by Overloading `__bool__`"_](https://www.philipzucker.com/overload_bool/),
but explores all branches exhaustively instead of randomly and fleshes out the concepts towards solving [Python Programming Puzzles](https://github.com/microsoft/PythonProgrammingPuzzles).

The solver incorporates heuristics from LLMs in addition to symbolic execution.

## Setup

### Install dependencies

- `python` with support for `pip` (e.g. `conda`), tested with Python 3.12
- `z3` or `cvc5` or both -- on mac with Homebrew, can install with `brew install z3 cvc5`
  
### Clone recursive

```
git clone --recursive https://github.com/namin/holey.git
```

### Setup environment
```
conda create -n holey python=3.12
conda activate holey
pip install -e ".[test,ollama,anthropic]"
```

## Run

### Help reference

```
python puzzle_solver.py --help
```

### Sanity check

```
python puzzle_solver.py --name-prefix HelloWorld:0
```

### Run all puzzles, saving stdout/stderr to file results.txt

```
python puzzle_solver.py  >results.txt 2>&
```

### Fallback to LLMs

Set `ANTHROPIC_API_KEY` for Claude or default to local Ollama.

```
python puzzle_solver.py --name-prefix ListIn:1  --llm
```

## Current status

The symbolic execution alone currently solves:
- 53% (192 out of 360) of `int` puzzles,
- 35% (253 out of 723) of `int` and `str` puzzles.

We have errors of all kinds still:
- 59 puzzles time out after 3 seconds at staging time (while building the SMTLIB program),
- 220 puzzles have an error during staging time,
- 70 generated SMTLIB programs return `sat` but the solution doesn't verify,
- 121 generated SMTLIB programs return non-`sat`, such as `unsat`, `unknown` or time out after 2 seconds.
- 1715-723=992 puzzles are not yet run, because their answer type is not `int` or `str`, such as `float`, `list` (of various specializations), etc.

See a detailed [stdout log](log/results.txt) of the current run.

# Source map

`.`<br/>
`├──` [`README.md`](README.md)<br/>
`├──` [`benchmarks`](benchmarks)<br/>
`│   └──` [`PythonProgrammingPuzzles`](benchmarks/PythonProgrammingPuzzles) _benchmark added as `git` submodule_<br/>
`├──` [`holey`](holey)<br/>
`│   ├──` [`__init__.py`](holey/__init__.py)<br/>
`│   ├──` [`backend.py`](holey/backend.py) _backend to SMTLIB batch processes_<br/>
`│   ├──` [`core.py`](holey/core.py) _includes tracer, symbolic classes, ..._<br/>
`│   ├──` [`llm.py`](holey/llm.py) _support for LLM generation and code extraction_<br/>
`│   └──` [`preprocessor.py`](holey/preprocessor.py) _includes node transformer and sat driver_<br/>
`├──` [`log`](log)<br/>
`│   └──` [`results.txt`](log/results.txt) _example run_<br/>
`├──` [`puzzle_solver.py`](puzzle_solver.py) _main routine for benchmark solver_<br/>
`├──` [`pyproject.toml`](pyproject.toml)<br/>
`└──` [`tests`](tests)<br/>
`    └──` [`test_core.py`](tests/test_core.py) _ran with `python -m pytest`, basic and LLM-generated_<br/>

