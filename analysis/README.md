# Examples of Holey for Analysis

## Example with Counting Requirements

```bash
python puzzle_solver.py  --smtlib-backends z3 cvc5  --answer-types int str float --name-prefix Study_5:0
```

```python
def sat(li: List[int]):
  return all([li.count(i) == i for i in range(10)])
```

[`Study_5_0.smt`](Study_5_0.smt) times out with `z3` and `cvc5`.

[`Study_5_0_small.smt`](Study_5_0_small.smt) succeeds with `z3`, and also with `cvc5 --produce-model --fmf-fun`.

Logs: [`Study_5_0.txt`](Study_5_0.txt).

## Example with `in` constraints

```bash
python puzzle_solver.py  --smtlib-backends cvc5 --answer-types "List[int]" "List[str]" --name-prefix Study_8:0
```

```python
def sat(ls: List[str]):
  return ls[1234] in ls[1235] and ls[1234] != ls[1235]
```

The [large query](Study_8_0.smt) times out, but the [smaller query](Study_8_0_small.smt), using 3 and 5, succeeds with `cvc5 --produce-model --fmf-fun` (has a degenerate result with `z3`). 

Logs: [`Study_8_0.txt`](Study_8_0.txt).
