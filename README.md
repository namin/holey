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

The symbolic execution currently solves:
- 61% (218 out of 360) of `int` puzzles,
- 25% (92 out of 363) of `str` puzzles,
- 43% (310 out of 723) overall.

with the following errors:
- 5 timeouts after 3 seconds at staging time (while generating the SMTLIB program)
- 249 errors at at staging time
- 81 SMTLIB programs returning `sat` but the original `sat` function failing on synthesized model input,
- 106 SMTLIB programs returning non-`sat` (e.g. `unsat`, `unknown` or timing out after 2 seconds
timeouts after staging (while building the SMTLIB program), errors during staging time, the SMTLIB
- 992 (out of 1715) puzzles not yet even attempted because their type is not `int` or `str`, such as `float`, `list` (of various specialization), etc.

### Extrapolation
- 77 smaller problems tried
- 14 successes on smaller problem
- 8 successful extrapolations

#### Matrix
- claude  (extrapolate) _5_ 1 0 0 0 0 0 0 1 1 0 0 1 0 1
- claude   (end-to-end) _4_ 0 0 0 0 0 1 0 1 1 0 0 1 0 0
- claude       (SMTLIB) _0_ 0 0 0 0 0 0 0 0 0 0 0 0 0 0
- gemini  (extrapolate) _3_ 1 0 0 0 0 0 1 0 0 0 1 0 0 0
- gemini   (end-to-end) _2_ 0 0 0 0 1 0 0 0 0 0 0 1 0 0
- gemini       (SMTLIB) _0_ 0 0 0 0 0 0 0 0 0 0 0 0 0 0
- ollama  (extrapolate) _1_ 0 0 0 1 0 0 0 0 0 0 0 0 0 0
- ollama   (end-to-end) _1_ 0 0 1 0 0 0 0 0 0 0 0 0 0 0
- ollama       (SMTLIB) _0_ 0 0 0 0 0 0 0 0 0 0 0 0 0 0

## Source map

`.`<br/>
`├──` [`README.md`](README.md)<br/>
`├──` [`benchmarks`](benchmarks)<br/>
`│   └──` [`PythonProgrammingPuzzles`](https://github.com/microsoft/PythonProgrammingPuzzles) _benchmark added as `git` submodule_<br/>
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

## Contribute!

I need help in completely fleshing out the symbolic executor as well as designing and implementing LLM-based heuristics to complement it.
See the [contributing guidelines](CONTRIBUTING.md), in particular discussing a workflow to find and fix issues driven by the benchmarks.
