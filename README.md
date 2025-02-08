# Holey

A Python library for program synthesis and symbolic execution that combines Z3's constraint solving with LLM-guided synthesis. Put holes in your Python code and let `holey` fill them using formal constraints, natural language specifications, or both.

Based on Philip Zucker's blog post ["Symbolic Execution by Overloading __bool__"](https://www.philipzucker.com/overload_bool/),
but explores all branches exhaustively instead of randomly.
