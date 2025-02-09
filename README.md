# Holey

A Python library for program synthesis and symbolic execution that combines Z3's constraint solving with LLM-guided synthesis. Put holes in your Python code and let `holey` fill them using formal constraints, natural language specifications, or both.

The symbolic execution is
based on Philip Zucker's blog post [_"Symbolic Execution by Overloading `__bool__`"_](https://www.philipzucker.com/overload_bool/),
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
