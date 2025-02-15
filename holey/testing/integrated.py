"""
Integration module that ties together symbolic execution, LLM solving, and analysis.
"""

from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass
from collections import defaultdict
import time
import logging
from pathlib import Path
from func_timeout import func_timeout, FunctionTimedOut

from ..core import SymbolicTracer
from ..llm import generate as llm_generate
from ..preprocessor import driver as drive_sat
from .systematic import TestCase, SolverAttempt
from .analysis import ResultAnalyzer

@dataclass
class RunConfig:
    """Configuration for a test run"""
    max_attempts: int = 3
    symbolic_timeout: int = 5
    llm_timeout: int = 10
    use_hybrid: bool = True
    record_smt: bool = True
    record_llm: bool = True

class IntegratedSolver:
    def __init__(self, config: Optional[RunConfig] = None):
        self.config = config or RunConfig()
        self.logger = logging.getLogger(__name__)
        self.smt_queries = []
        self.llm_queries = []

    def solve_case(self, case: TestCase) -> Dict[str, List[SolverAttempt]]:
        """Solve a single test case using all available strategies"""
        results = defaultdict(list)
        
        # Try each strategy
        for strategy in ['symbolic', 'llm', 'hybrid']:
            for attempt in range(self.config.max_attempts):
                self.logger.info(f"Strategy {strategy}, attempt {attempt + 1}")
                
                start_time = time.time()
                try:
                    if strategy == 'symbolic':
                        solution = self._symbolic_solve(case)
                    elif strategy == 'llm':
                        solution = self._llm_solve(case)
                    else:  # hybrid
                        solution = self._hybrid_solve(case)
                        
                    time_taken = time.time() - start_time
                    
                    if solution is not None and self._verify_solution(solution, case):
                        results[strategy].append(
                            SolverAttempt(
                                solution=solution,
                                time_taken=time_taken,
                                success=True
                            )
                        )
                    else:
                        results[strategy].append(
                            SolverAttempt(
                                solution=None,
                                time_taken=time_taken,
                                success=False,
                                error="Solution verification failed"
                            )
                        )
                except Exception as e:
                    time_taken = time.time() - start_time
                    results[strategy].append(
                        SolverAttempt(
                            solution=None,
                            time_taken=time_taken,
                            success=False,
                            error=str(e)
                        )
                    )
                    
        return results

    def _symbolic_solve(self, case: TestCase) -> Optional[Any]:
        """Try solving with symbolic execution"""
        try:
            result = func_timeout(
                self.config.symbolic_timeout,
                self._run_symbolic,
                args=(case,)
            )
            return result
        except FunctionTimedOut:
            self.logger.warning("Symbolic execution timed out")
            return None
        except Exception as e:
            self.logger.error(f"Symbolic execution failed: {e}")
            return None

    def _run_symbolic(self, case: TestCase) -> Optional[Any]:
        """Run symbolic execution with full error capture"""
        typ = None
        if case.ans_type == 'int':
            typ = int
        elif case.ans_type == 'str':
            typ = str
        else:
            raise ValueError(f"Unsupported answer type: {case.ans_type}")

        try:
            sym_var = drive_sat(case.sat_func, typ, None)
            tracer = sym_var.tracer
            solution = tracer.solution()
            
            if solution is None:
                return None
                
            solution_var = tracer.solution_var(solution, sym_var)
            if solution_var is None:
                return None
                
            return typ(str(solution_var))
        except Exception as e:
            self.logger.error(f"Error in symbolic execution: {e}")
            raise

    def _llm_solve(self, case: TestCase) -> Optional[Any]:
        """Try solving with LLM"""
        prompt = self._build_llm_prompt(case)
        
        try:
            result = func_timeout(
                self.config.llm_timeout,
                llm_generate,
                args=(prompt,),
                kwargs={'temperature': 0.3}
            )
            
            if self.config.record_llm:
                self.llm_queries.append({
                    'prompt': prompt,
                    'response': result
                })
                
            return self._parse_llm_response(result, case.ans_type)
        except FunctionTimedOut:
            self.logger.warning("LLM solving timed out")
            return None
        except Exception as e:
            self.logger.error(f"LLM solving failed: {e}")
            return None

    def _build_llm_prompt(self, case: TestCase) -> str:
        """Build prompt for LLM solver"""
        return f"""Solve this programming puzzle:
Function to satisfy:
{case.sat_func}

Required return type: {case.ans_type}

Consider:
1. The function must return True for your solution
2. Your solution must be a valid {case.ans_type}
3. Pay attention to any constraints in the function

Return only the Python constant that satisfies the function."""

    def _parse_llm_response(self, response: str, ans_type: str) -> Optional[Any]:
        """Parse and validate LLM response"""
        response = response.strip()
        
        # Remove any code block markers
        if response.startswith('```'):
            response = response.split('\n')[-2].strip()
        
        # Remove any quotes for string values
        if ans_type == 'str' and (response.startswith('"') or response.startswith("'")):
            response = response[1:-1]
            
        # Handle integer expressions
        if ans_type == 'int':
            try:
                # Safely evaluate simple mathematical expressions
                if any(op in response for op in ['*', '**']):
                    return eval(response)
                return int(response)
            except Exception as e:
                self.logger.error(f"Failed to parse int response: {e}")
                return None
                
        return response

    def _hybrid_solve(self, case: TestCase) -> Optional[Any]:
        """Try hybrid solving approach"""
        # First try symbolic execution
        solution = self._symbolic_solve(case)
        if solution is not None:
            return solution
            
        # If symbolic fails, use its insights for LLM
        symbolic_insights = self._extract_symbolic_insights()
        prompt = f"""Previous symbolic execution failed for this puzzle:
{case.sat_func}

Solver insights:
{symbolic_insights}

Can you propose a solution that:
1. Satisfies the core constraints
2. Considers the symbolic execution's attempts
3. Might be simpler than what the solver tried

Return only the Python constant that satisfies the function."""

        if self.config.record_llm:
            self.llm_queries.append({
                'prompt': prompt,
                'context': 'hybrid'
            })
            
        try:
            result = func_timeout(
                self.config.llm_timeout,
                llm_generate,
                args=(prompt,),
                kwargs={'temperature': 0.5}
            )
            return self._parse_llm_response(result, case.ans_type)
        except Exception as e:
            self.logger.error(f"Hybrid solving failed: {e}")
            return None

    def _extract_symbolic_insights(self) -> str:
        """Extract insights from symbolic execution attempts"""
        if not self.smt_queries:
            return "No symbolic execution data available"
            
        # Get last few queries
        recent_queries = self.smt_queries[-3:]
        
        insights = []
        for query in recent_queries:
            # Extract key constraints
            constraints = [line for line in query.split('\n') 
                         if line.strip().startswith('(assert')]
            insights.extend(constraints)
            
        return "\n".join(insights)

    def _verify_solution(self, solution: Any, case: TestCase) -> bool:
        """Verify if a solution satisfies the test case"""
        try:
            namespace = {}
            exec(case.sat_func, namespace)
            sat = namespace['sat']
            return sat(solution)
        except Exception as e:
            self.logger.error(f"Verification error: {e}")
            return False

    def analyze_run(self, results: Dict[str, List[SolverAttempt]]) -> Dict[str, Any]:
        """Analyze results from a test run"""
        analyzer = ResultAnalyzer(results)
        return analyzer.analyze_strategy_performance()

    def get_recommendation(self, case: TestCase, results: Dict[str, Any]) -> str:
        """Get strategy recommendation for similar cases"""
        if not results:
            return "No data available for recommendations"
            
        best_strategy = max(
            results.items(),
            key=lambda x: (
                x[1]['success_rate'],
                -x[1]['avg_time']  # Break ties with speed
            )
        )[0]
        
        return f"Recommended strategy for similar cases: {best_strategy}"