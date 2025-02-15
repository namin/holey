from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Callable
import json
import time
from collections import defaultdict
from ..preprocessor import driver as drive_sat
from ..core import SymbolicTracer
from ..llm import generate as llm_generate

@dataclass
class TestCase:
    sat_func: str
    ans_type: str
    name: str
    expected: Optional[Any] = None
    metadata: Dict[str, Any] = None
    difficulty: Optional[float] = None

@dataclass
class SolverAttempt:
    solution: Any
    time_taken: float
    success: bool
    error: Optional[str] = None
    smt_queries: List[str] = None
    llm_queries: List[Dict] = None

@dataclass
class TestResult:
    testcase: TestCase
    attempts: List[SolverAttempt]
    strategy: str
    
    @property
    def success_rate(self) -> float:
        return sum(1 for a in self.attempts if a.success) / len(self.attempts)
    
    @property
    def avg_time(self) -> float:
        return sum(a.time_taken for a in self.attempts) / len(self.attempts)

class SystematicTester:
    def __init__(self, solver, attempts_per_case: int = 10):
        self.solver = solver
        self.attempts_per_case = attempts_per_case
        self.results: List[TestResult] = []
        self.strategy_stats = defaultdict(list)
        self.smt_queries = []
        self.llm_queries = []

    def _verify_solution(self, solution: Any, test_case: TestCase) -> bool:
        """Verify if a solution satisfies the test case"""
        try:
            namespace = {}
            exec(test_case.sat_func, namespace)
            sat = namespace['sat']
            return sat(solution)
        except Exception as e:
            print(f"Verification error: {e}")
            return False

    def symbolic_solve(self, test_case: TestCase) -> Optional[Any]:
        """Use symbolic execution to solve a test case"""
        typ = None
        if test_case.ans_type == 'int':
            typ = int
        elif test_case.ans_type == 'str':
            typ = str
        else:
            raise ValueError(f"Unsupported answer type: {test_case.ans_type}")

        def smt_callback(query: str):
            self.smt_queries.append(query)

        try:
            sym_var = drive_sat(test_case.sat_func, typ, None)
            tracer = sym_var.tracer
            solution = tracer.solution()
            
            if solution is None:
                return None
                
            solution_var = tracer.solution_var(solution, sym_var)
            if solution_var is None:
                return None
                
            return typ(str(solution_var))
        except Exception as e:
            print(f"Symbolic execution error: {e}")
            return None

    def llm_solve(self, test_case: TestCase) -> Optional[Any]:
        """Use LLM to solve a test case"""
        prompt = f"""Solve this programming puzzle:
Function to satisfy:
{test_case.sat_func}

Required return type: {test_case.ans_type}

Consider:
1. The function must return True for your solution
2. Your solution must be a valid {test_case.ans_type}
3. Pay attention to any constraints in the function

Return only the Python constant that satisfies the function."""

        self.llm_queries.append({"prompt": prompt})

        try:
            result = llm_generate(prompt, temperature=0.3)
            if not result:
                return None

            # Extract and verify solution
            if test_case.ans_type == 'str':
                solution = result.strip().strip("'\"")
            elif test_case.ans_type == 'int':
                solution = int(result.strip())
            else:
                return None

            if self._verify_solution(solution, test_case):
                return solution
            return None
        except Exception as e:
            print(f"LLM solving error: {e}")
            return None

    def hybrid_solve(self, test_case: TestCase) -> Optional[Any]:
        """Try both symbolic and LLM approaches with feedback loop"""
        # First try symbolic execution
        sym_solution = self.symbolic_solve(test_case)
        if sym_solution is not None:
            return sym_solution

        # If symbolic fails, use its insights for LLM
        prompt = f"""Previous symbolic execution failed for this puzzle:
{test_case.sat_func}

SMT queries attempted:
{self.smt_queries[-3:] if self.smt_queries else 'None'}

Can you propose a solution that:
1. Satisfies the core constraints
2. Considers the symbolic execution's attempts
3. Might be simpler than what the solver tried

Return only the Python constant that satisfies the function."""

        self.llm_queries.append({"prompt": prompt, "context": "hybrid"})
        
        try:
            result = llm_generate(prompt, temperature=0.5)
            if not result:
                return None

            if test_case.ans_type == 'str':
                solution = result.strip().strip("'\"")
            elif test_case.ans_type == 'int':
                solution = int(result.strip())
            else:
                return None

            if self._verify_solution(solution, test_case):
                return solution
            return None
        except Exception as e:
            print(f"Hybrid solving error: {e}")
            return None

    def run_reliability_test(self, test_cases: List[TestCase]) -> Dict[str, Any]:
        """Run comprehensive reliability testing"""
        results = defaultdict(list)
        
        for case in test_cases:
            print(f"\nTesting {case.name}")
            case_results = {
                'symbolic': [],
                'llm': [],
                'hybrid': []
            }
            
            for attempt in range(self.attempts_per_case):
                print(f"  Attempt {attempt + 1}/{self.attempts_per_case}")
                
                # Try each strategy
                for strategy in ['symbolic', 'llm', 'hybrid']:
                    start_time = time.time()
                    try:
                        if strategy == 'symbolic':
                            solution = self.symbolic_solve(case)
                        elif strategy == 'llm':
                            solution = self.llm_solve(case)
                        else:  # hybrid
                            solution = self.hybrid_solve(case)
                            
                        time_taken = time.time() - start_time
                        
                        if solution is not None and self._verify_solution(solution, case):
                            case_results[strategy].append(
                                SolverAttempt(
                                    solution=solution,
                                    time_taken=time_taken,
                                    success=True
                                )
                            )
                        else:
                            case_results[strategy].append(
                                SolverAttempt(
                                    solution=None,
                                    time_taken=time_taken,
                                    success=False,
                                    error="Solution verification failed"
                                )
                            )
                    except Exception as e:
                        time_taken = time.time() - start_time
                        case_results[strategy].append(
                            SolverAttempt(
                                solution=None,
                                time_taken=time_taken,
                                success=False,
                                error=str(e)
                            )
                        )
            
            # Record results for this case
            for strategy, attempts in case_results.items():
                results[strategy].extend(attempts)
                self.results.append(
                    TestResult(
                        testcase=case,
                        attempts=attempts,
                        strategy=strategy
                    )
                )
                
        return self._analyze_results(results)

    def _analyze_results(self, results: Dict[str, List[SolverAttempt]]) -> Dict[str, Any]:
        """Detailed analysis of test results"""
        analysis = {
            "overall": {},
            "by_strategy": {},
            "by_type": {},
            "failure_patterns": {},
            "recommendations": []
        }
        
        # Overall statistics
        for strategy, attempts in results.items():
            success_rate = sum(1 for a in attempts if a.success) / len(attempts)
            avg_time = sum(a.time_taken for a in attempts) / len(attempts)
            
            analysis["by_strategy"][strategy] = {
                "success_rate": success_rate,
                "avg_time": avg_time,
                "total_attempts": len(attempts)
            }
            
        # Analyze failure patterns
        failure_patterns = defaultdict(int)
        for attempts in results.values():
            for attempt in attempts:
                if not attempt.success and attempt.error:
                    failure_patterns[attempt.error] += 1
                    
        analysis["failure_patterns"] = dict(failure_patterns)
        
        # Generate recommendations
        analysis["recommendations"] = self._generate_recommendations(analysis)
        
        return analysis

    def _generate_recommendations(self, analysis: Dict) -> List[str]:
        """Generate improvement recommendations based on analysis"""
        recommendations = []
        
        # Check for systematic failures
        for strategy, stats in analysis["by_strategy"].items():
            if stats["success_rate"] < 0.5:
                recommendations.append(
                    f"Strategy '{strategy}' needs improvement (success rate: {stats['success_rate']:.2f})"
                )
                
        # Check for slow strategies
        for strategy, stats in analysis["by_strategy"].items():
            if stats["avg_time"] > 5.0:  # 5 seconds threshold
                recommendations.append(
                    f"Strategy '{strategy}' is slow (avg time: {stats['avg_time']:.2f}s)"
                )
                
        # Analyze failure patterns
        common_failures = sorted(
            analysis["failure_patterns"].items(),
            key=lambda x: x[1],
            reverse=True
        )[:3]
        
        for error, count in common_failures:
            recommendations.append(
                f"Common failure: {error} (occurred {count} times)"
            )
            
        return recommendations

    def generate_report(self) -> str:
        """Generate detailed test report"""
        report = ["# Systematic Testing Report\n"]
        
        # Overall statistics
        total_cases = len(self.results)
        successful_cases = sum(1 for r in self.results if r.success_rate > 0)
        
        report.append(f"## Overall Statistics")
        report.append(f"- Total test cases: {total_cases}")
        report.append(f"- Successful cases: {successful_cases}")
        report.append(f"- Overall success rate: {successful_cases/total_cases:.2%}\n")
        
        # Strategy performance
        report.append("## Strategy Performance")
        for strategy, stats in self.strategy_stats.items():
            success_rate = sum(1 for s in stats if s)/len(stats)
            report.append(f"### {strategy}")
            report.append(f"- Success rate: {success_rate:.2%}")
            report.append(f"- Total attempts: {len(stats)}\n")
            
        # Failure analysis
        report.append("## Failure Analysis")
        failure_counts = defaultdict(int)
        for result in self.results:
            for attempt in result.attempts:
                if not attempt.success and attempt.error:
                    failure_counts[attempt.error] += 1
                    
        for error, count in sorted(failure_counts.items(), key=lambda x: x[1], reverse=True):
            report.append(f"- {error}: {count} occurrences")
            
        return "\n".join(report)