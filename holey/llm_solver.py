from .llm import generate as llm_generate, extract_code_blocks
from .backend import run_smt
from typing import List, Any, Dict, Optional, Tuple

class LLMSolver:
    """LLM-assisted solving capabilities that can be hooked into SymbolicTracer"""
    
    def __init__(self, temperature=0.3):
        self.temperature = temperature
        self.context = {}  # Store problem context and history
        self.cache = {}  # Cache LLM responses

    def _parse_llm_response(self, response: str, ans_type: str) -> Optional[Any]:
        """Parse LLM response by evaluating as Python"""
        # Clean the response
        response = response.strip()
        if '```' in response:
            # Extract from code block if present
            response = response.split('```')[1]
            if 'python' in response:
                response = response.split('\n', 1)[1]
            response = response.strip('`').strip()

        try:
            # Let Python do the work
            result = eval(response)
            # Verify type matches
            if ans_type == 'str' and isinstance(result, str):
                return result
            elif ans_type == 'int' and isinstance(result, int):
                return result
        except:
            pass
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
            
            parsed = self._parse_llm_response(result, ans_type)
            if parsed is None:
                print('LLM returned unparseable response', result)
                continue
                
            if not check_result(parsed, sat_func):
                print('LLM result fails to verify for puzzle '+name)
            else:
                print('LLM result verifies for puzzle '+name)
                return parsed
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