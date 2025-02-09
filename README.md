# Holey

A Python library for program synthesis and symbolic execution that combines Z3's constraint solving with LLM-guided synthesis. Put holes in your Python code and let `holey` fill them using formal constraints, natural language specifications, or both.

The symbolic execution is
based on Philip Zucker's blog post [_"Symbolic Execution by Overloading `__bool__`"_](https://www.philipzucker.com/overload_bool/),
but explores all branches exhaustively instead of randomly and fleshes out the concepts towards solving [Python Programming Puzzles](https://github.com/microsoft/PythonProgrammingPuzzles).

The solver incorporates heuristics from LLMs in addition to symbolic execution.
