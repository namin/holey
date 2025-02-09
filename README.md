# Holey

A Python library for program synthesis and symbolic execution that combines Z3's constraint solving with LLM-guided synthesis. Put holes in your Python code and let `holey` fill them using formal constraints, natural language specifications, or both.

The symbolic execution is
inspired by Philip Zucker's blog post [_"Symbolic Execution by Overloading `__bool__`"_](https://www.philipzucker.com/overload_bool/),
but explores all branches exhaustively instead of randomly and fleshes out the concepts towards solving [Python Programming Puzzles](https://github.com/microsoft/PythonProgrammingPuzzles).

The solver incorporates heuristics from LLMs in addition to symbolic execution.

## Setup

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
python puzzle_solver.py  >results_wip.txt 2>&
```

### Fallback to LLMs

Set `ANTHROPIC_API_KEY` for Claude or default to local Ollama.

```
python puzzle_solver.py --name-prefix ListIn:0  --llm
```

## Current status

The symbolic execution alone currently solves:
- 53% (192 out of 360) of `int` puzzles,
- 34% (243 out of 723) of `int` and `str` puzzles.

We have errors of all kinds still:
- 59 puzzles timeout after 3 seconds at staging time (while building the SMTLIB program),
- 71 generated SMTLIB programs return `sat` but the solution doesn't verify (so code generation is buggy),
- 130 generated SMTLIB programs return non-`sat`, such as `unsat`, `unknown` or time out after 2 seconds.
- 1715-723=992 puzzles are not yet run, because their answer type is not `int` or `str`, such as `float`, `list` (of various specializations), etc.

# Source map

```
.
├── README.md
├── benchmarks
│   └── PythonProgrammingPuzzles (benchmark added as git submodule)
├── holey
│   ├── __init__.py
│   ├── backends
│   │   ├── __init__.py
│   │   ├── base.py (stub)
│   │   ├── cvc5_backend.py (unusable because cvc5 bindings segfaults)
│   │   ├── mock_backend.py (! main backend)
│   │   └── z3_backend.py (unusable because z3 bindings segfaults)
│   ├── core.py (! includes tracer, symbolic classes, ... !)
│   ├── llm.py (! support for LLM generation and code extraction !)
│   └── preprocessor.py (! includes node transformer and sat driver !)
├── puzzle_solver.py (! main routine for benchmark solver !)
├── pyproject.toml
└── tests
    └── test_core.py (ran with python -m pytest, LLM-generated, not much there)
```
