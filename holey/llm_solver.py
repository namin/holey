from .llm import generate as llm_generate

class LLMSolver:
    """LLM-assisted solving capabilities that can be hooked into SymbolicTracer"""
    
    def __init__(self, temperature=0.3):
        self.temperature = temperature
        self.context = {}  # Store problem context and history
        self.cache = {}  # Cache LLM responses
        
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
            
    def get_constraint_refinements(self, constraints, attempt=0):
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
            return [parse_smtlib(s) for s in suggestions]
        except:
            return []
            
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
