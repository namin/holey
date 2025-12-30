# Contributing guide for `namin/holey`

See [some guidelines and cheatsheet for git and GitHub](https://namin.org/git).

## Workflow to find and fix an issue

See `log/results.txt`

This file is generated with the command

```
PYTHONHASHSEED=0 python puzzle_solver.py  --smtlib-backends z3 cvc5 --answer-types int str float 'List[int]' 'List[str]' >log/results.txt     
```

Focus on a puzzle with an error, debug it and propose a fix.

Re-generate the entire `log/results.txt` file and make sure there are only improvements by diffing, looking at the final stats in particular. As you check in the updated run log, also update the final stats in the `README.md`.

For quick testing, you can use _`--name-prefix PUZZLE_NAME_OR_PREFIX`_ with the `python_solver.py` to focus on some puzzles.

Send a GitHub pull request from your fork and branch to the `main` branch on `namin/holey`.
