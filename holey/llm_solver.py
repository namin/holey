from .llm import generate as llm_generate, extract_code_blocks
from .backend import run_smt
from typing import List, Any, Dict, Optional, Tuple

class LLMSolver:
    """LLM-assisted solving capabilities that can be hooked into SymbolicTracer"""
    
    def __init__(self, temperature=0.3):
        self.temperature = temperature
        self.context = {}  # Store problem context and history
        self.cache = {}  # Cache LLM responses

    def extrapolate(self, puzzle_small, puzzle_large, reason, result_small, check_result, cmds=None):
        print('Extrapolating...')
        sat_func_small = puzzle_small['sat_function']
        sat_func_large = puzzle_large.get('sat_function', puzzle_large.get('sat', ''))
        prompt = f"""Given the smaller satisfiability predicate:
```python
{sat_func_small}
```
a result is `{result_small}`, then what is a result for the bigger satisfiability predicate:
```python
{sat_func_large}
```
?
Answer with just a Python expression that evaluates to the bigger result.
"""
        response = llm_generate(prompt)
        print(response)
        return None

    def smtlib_solve(self, sat_func: str, ans_type: str, name: str, log: str, check_result, cmds=None) -> Optional[str]:
        print('Asking LLM for SMTLIB')
        prompt = f"""Return a modified SMTLIB z3 program that captures the intent of the `sat` function of puzzle {name}:
{sat_func}

This is the log, you may copy most of any SMTLIB program below.
{log}

Return only the new SMTLIB program without any context.
"""
        blocks = extract_code_blocks(llm_generate(prompt))
        model = None
        result = None
        flag = None
        for smt in blocks:
            flag, model = run_smt(smt, cmds)
            if flag == "sat":
                break
        if model:
            llm_result = model['x']
            if check_result(llm_result, sat_func):
                print("LLM result confirmed for puzzle " + name)
                result = llm_result
        return None

    def solve_end2end(self, sat_func: str, ans_type: str, name: str, check_result) -> Optional[str]:
        print('Asking LLM for whole answer')
        prompt = f"""Return a constant Python value of type {ans_type} to solve puzzle {name}, where your goal is to synthesize the first argument that makes this `sat` function return `True`:
{sat_func}

Return only the Python constant without any context.
"""
        results = extract_code_blocks(llm_generate(prompt))
        for result in results:
            print('LLM result', result)
            if ans_type == 'int':
                try:
                    result = int(result)
                except ValueError as e:
                    print('LLM returned bad type for int', e)
                    break
            elif ans_type == 'str':
                if result and result[0] in ["'", '"']:
                    result = result[1:-1] # TODO
            if not check_result(result, sat_func):
                print('LLM result fails to verify for puzzle '+name)
            else:
                print('LLM result verifies for puzzle '+name)
                return result
        return None
        
    def get_branch_guidance(self, condition, path_conditions):
        """Get LLM guidance on which branch to take"""
        cache_key = (str(condition), str(path_conditions))
        if cache_key in self.cache:
            return self.cache[cache_key]
            
        prompt = f"""Given the current symbolic execution state:
Path conditions: {path_conditions}
Current branch condition: {condition}

Should we take the True or False branch? Consider:
1) Which branch is more likely to lead to a solution?
2) What patterns suggest the better path?
3) How do the path conditions constrain viable solutions?

Return only True or False without explanation."""

        try:
            result = llm_generate(prompt, 
                                temperature=self.temperature).strip().lower() == 'true'
            self.cache[cache_key] = result
            return result
        except Exception as e:
            return True  # Default to original behavior
            
    def add_constraint_refinements(self, constraints, backend, attempt=0):
        """Get LLM suggestions for refining constraints"""
        prompt = f"""These SMT constraints are challenging to solve:
{constraints}

Previous attempts: {attempt}

Suggest ways to:
1) Break into simpler sub-constraints
2) Add helpful intermediate assertions
3) Rewrite in more solver-friendly form
4) Add bounds or range restrictions

Return only SMT-LIB constraints without explanation."""

        try:
            suggestions = extract_code_blocks(
                llm_generate(prompt, 
                           temperature=self.temperature)
            )
        except:
            suggestions = []
        backend.solver.add_text("\n".join(suggestions))
            
    def get_pattern_guidance(self, examples, problem_type=None):
        """Get LLM insights from solved examples"""
        prompt = f"""Based on these solved examples:
{examples}

Problem type: {problem_type or 'Unknown'}

Identify:
1) Common solution patterns
2) Key intermediate steps 
3) Useful transformations
4) Bounds and constraints to add

Return only SMT-LIB patterns without explanation."""

        try:
            return extract_code_blocks(
                llm_generate(prompt,
                           temperature=self.temperature)
            )
        except:
            return []
