# Holey

A Python library for program synthesis and symbolic execution that combines SMT (Z3, CVC5, ...) constraint solving with LLM-guided synthesis. Put holes in your Python code and let `holey` fill them using formal constraints, natural language specifications, or both.

The symbolic execution is
inspired by Philip Zucker's blog post [_"Symbolic Execution by Overloading `__bool__`"_](https://www.philipzucker.com/overload_bool/),
but explores all branches exhaustively instead of randomly and fleshes out the concepts towards solving [Python Programming Puzzles](https://github.com/microsoft/PythonProgrammingPuzzles).

The solver incorporates heuristics from LLMs in addition to symbolic execution.

- [Diagram: multi-stage programming SMT: overloaded execution generates constraints](https://github.com/namin/holey/blob/main/diagram_staging_SMT.md)
- [Diagram: overview](https://github.com/namin/holey/blob/main/diagram.md)
- [![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/namin/holey)

## Setup

### Install dependencies

- `python` with support for `pip` (e.g. `conda`), tested with Python 3.12
- `z3` or `cvc5` or both -- on mac with Homebrew, can install with `brew install z3 cvc5`
  
### Clone recursive

```
git clone --recursive https://github.com/namin/holey.git
```

### Setup Python environment
```
conda create -n holey python=3.12
conda activate holey
pip install -e ".[test,ollama,anthropic]"
```

### Setup LLM environments

For each LLM you want to use, provide an LLM API key, even if only a dummy one is needed.
Only provide a key if you want to use that particular LLM provider.
All provided keys will be used in parallel to generate a matrix of successes per LLM provider.

```
export OLLAMA_API_KEY=ollama
export ANTHROPIC_API_KEY=...
export GEMINI_API_KEY=...
export OPENAI_API_KEY=...
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

```
python puzzle_solver.py --name-prefix ListIn:1  --llm
```

## Current status

The symbolic execution alone currently solves:
- 65% (235 out of 360) of `int` puzzles,
- 44% (160 out of 363) of `str` puzzles,
- 18% (106 out of 591) of `List[int]` puzzles,
- 22% (31 out of 141) of `List[str]` puzzles,
- 53% (27 out of 51) of `float` puzzles,
- 37% (559 out of 1506) overall.

with the following errors:
- 62 timeouts after 3 seconds at staging time (while generating the SMTLIB program)
- 304 errors at staging time
- 30 SMTLIB programs returning `sat` but the original `sat` function failing on synthesized model input,
- 563 SMTLIB programs returning non-`sat` (e.g. `unsat`, `unknown` or timing out after 2 seconds)
- 209 (out of 1715) puzzles not yet even attempted because their type is not `int` or `str`, such as `float`, `list` (of various specialization), etc.

### Extrapolation
- 208 smaller problems tried
- 48 successes on smaller problem

### Earlier extrapolation results
- 123 smaller problems tried
- 11 successes on smaller problem
- 9 successful extrapolations

#### Extrapolated puzzles
Study_1:0 PandigitalSquare:0 CircularShiftNum:2 WeirdDecodeVowels:0 TripleDouble:0 MaxDelta:0 MinConsecutiveSum:2 MaxConsecutiveSum:0 BirthdayParadox:0 BirthdayParadox:1 Tutorial5:0
#### Successfully extrapolated puzzles
Study_1:0 PandigitalSquare:0 WeirdDecodeVowels:0 TripleDouble:0 MaxDelta:0 MinConsecutiveSum:2 MaxConsecutiveSum:0 BirthdayParadox:0 Tutorial5:0

#### Matrix
- claude      (extrapolate) _8_ 1 1 0 1 1 0 1 1 1 0 1
- claude       (end-to-end) _5_ 1 1 0 1 0 0 0 0 1 1 0
- claude           (SMTLIB) _0_ 0 0 0 0 0 0 0 0 0 0 0
- gemini      (extrapolate) _5_ 0 1 0 1 1 1 1 0 0 0 0
- gemini       (end-to-end) _5_ 0 0 0 0 1 1 0 1 1 0 1
- gemini           (SMTLIB) _0_ 0 0 0 0 0 0 0 0 0 0 0
- ollama      (extrapolate) _2_ 0 0 0 0 0 0 1 0 0 0 1
- ollama       (end-to-end) _2_ 0 0 0 0 0 0 0 0 1 0 1
- ollama           (SMTLIB) _0_ 0 0 0 0 0 0 0 0 0 0 0

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
