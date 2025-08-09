# Examples of Holey for Analysis

## Example with Counting Requirements

```bash
python puzzle_solver.py  --smtlib-backends z3 cvc5  --answer-types int str float --name-prefix Study_5:0
```

```python
def sat(li: List[int]):
  return all([li.count(i) == i for i in range(10)])
```

[`Study_5:0.smt`](Study_5:0.smt) times out with `z3` and `cvc5`.

[`Study_5:0_small.smt`](Study_5.0_small.smt) succeeds with `z3`, and also with `cvc5 --produce-model --fmf-fun`.

