"""
Test driver that ties together all components of the testing framework.
Provides high-level interface for running systematic tests.
"""

from typing import Optional, List, Dict, Any
from pathlib import Path
import json
import logging
from datetime import datetime

from .systematic import TestCase, SystematicTester
from .solvers import ProgressiveSolver
from .analysis import ResultAnalyzer

class TestDriver:
    def __init__(self, 
                 output_dir: str = 'test_output',
                 log_level: int = logging.INFO):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Setup logging
        self.setup_logging(log_level)
        
        self.solver = ProgressiveSolver()
        self.tester = SystematicTester(self.solver)
        
        # Track test history
        self.history_file = self.output_dir / 'test_history.json'
        self.history = self.load_history()

    def setup_logging(self, log_level: int):
        """Setup logging configuration"""
        log_file = self.output_dir / f'test_run_{datetime.now():%Y%m%d_%H%M%S}.log'
        
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        
        self.logger = logging.getLogger(__name__)

    def load_history(self) -> Dict[str, Any]:
        """Load test history from file"""
        if self.history_file.exists():
            with open(self.history_file) as f:
                return json.load(f)
        return {'runs': []}

    def save_history(self):
        """Save test history to file"""
        with open(self.history_file, 'w') as f:
            json.dump(self.history, f, indent=2)

    def run_test_suite(self, 
                      test_cases: List[TestCase],
                      tag: Optional[str] = None) -> Dict[str, Any]:
        """Run a complete test suite"""
        self.logger.info(f"Starting test run with {len(test_cases)} cases")
        
        run_id = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        try:
            # Run tests
            results = self.tester.run_reliability_test(test_cases)
            
            # Analyze results
            analyzer = ResultAnalyzer(results)
            analysis_dir = self.output_dir / f'analysis_{run_id}'
            analyzer.save_analysis(analysis_dir)
            
            # Generate report
            report = analyzer.generate_report()
            report_file = analysis_dir / 'report.md'
            report_file.write_text(report)
            
            # Update history
            run_record = {
                'id': run_id,
                'timestamp': datetime.now().isoformat(),
                'tag': tag,
                'num_cases': len(test_cases),
                'results': results,
                'analysis_dir': str(analysis_dir),
                'status': 'completed'
            }
            
            # Track improvements/regressions
            if self.history['runs']:
                last_run = self.history['runs'][-1]
                improvements = self._analyze_improvements(last_run['results'], results)
                run_record['improvements'] = improvements
            
            self.history['runs'].append(run_record)
            self.save_history()
            
            self.logger.info(f"Test run {run_id} completed successfully")
            return results
            
        except Exception as e:
            self.logger.error(f"Test run {run_id} failed: {e}", exc_info=True)
            
            # Record failure in history
            run_record = {
                'id': run_id,
                'timestamp': datetime.now().isoformat(),
                'tag': tag,
                'num_cases': len(test_cases),
                'error': str(e),
                'status': 'failed'
            }
            self.history['runs'].append(run_record)
            self.save_history()
            
            raise

    def _analyze_improvements(self, 
                            old_results: Dict[str, Any],
                            new_results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze improvements between test runs"""
        improvements = {
            'success_rate': {},
            'speed': {},
            'overall': 'neutral'
        }
        
        # Compare success rates
        for strategy in old_results['by_strategy']:
            if strategy in new_results['by_strategy']:
                old_rate = old_results['by_strategy'][strategy]['success_rate']
                new_rate = new_results['by_strategy'][strategy]['success_rate']
                
                change = new_rate - old_rate
                improvements['success_rate'][strategy] = {
                    'change': change,
                    'status': 'improved' if change > 0 else 'regressed' if change < 0 else 'unchanged'
                }
                
        # Compare execution times
        for strategy in old_results['by_strategy']:
            if strategy in new_results['by_strategy']:
                old_time = old_results['by_strategy'][strategy]['avg_time']
                new_time = new_results['by_strategy'][strategy]['avg_time']
                
                change = old_time - new_time  # Positive means faster
                improvements['speed'][strategy] = {
                    'change': change,
                    'status': 'improved' if change > 0 else 'regressed' if change < 0 else 'unchanged'
                }
                
        # Determine overall status
        improved_count = sum(1 for metric in improvements.values() 
                           if isinstance(metric, dict) and 
                           any(v['status'] == 'improved' for v in metric.values()))
        regressed_count = sum(1 for metric in improvements.values()
                            if isinstance(metric, dict) and 
                            any(v['status'] == 'regressed' for v in metric.values()))
                            
        if improved_count > regressed_count:
            improvements['overall'] = 'improved'
        elif regressed_count > improved_count:
            improvements['overall'] = 'regressed'
        
        return improvements

    def analyze_history(self) -> Dict[str, Any]:
        """Analyze test history for trends"""
        analysis = {
            'total_runs': len(self.history['runs']),
            'success_rate': {},
            'trends': {},
            'recommendations': []
        }
        
        # Calculate success rates over time
        for strategy in ['symbolic', 'llm', 'hybrid']:
            rates = []
            for run in self.history['runs']:
                if run['status'] == 'completed':
                    strategy_results = run['results']['by_strategy'].get(strategy, {})
                    rate = strategy_results.get('success_rate', 0)
                    rates.append(rate)
            
            if rates:
                analysis['success_rate'][strategy] = {
                    'current': rates[-1],
                    'trend': 'improving' if rates[-1] > rates[0] else 'declining',
                    'stability': self._calculate_stability(rates)
                }
                
        # Generate recommendations
        self._generate_historical_recommendations(analysis)
        
        return analysis

    def _calculate_stability(self, values: List[float]) -> str:
        """Calculate stability metric from a series of values"""
        if len(values) < 2:
            return 'unknown'
            
        variations = [abs(b - a) for a, b in zip(values[:-1], values[1:])]
        avg_variation = sum(variations) / len(variations)
        
        if avg_variation < 0.05:
            return 'very_stable'
        elif avg_variation < 0.1:
            return 'stable'
        elif avg_variation < 0.2:
            return 'unstable'
        else:
            return 'very_unstable'

    def _generate_historical_recommendations(self, analysis: Dict[str, Any]):
        """Generate recommendations based on historical analysis"""
        recommendations = []
        
        # Check for unstable strategies
        for strategy, stats in analysis['success_rate'].items():
            if stats['stability'] in ['unstable', 'very_unstable']:
                recommendations.append(
                    f"Strategy '{strategy}' shows high variability. Consider:"
                    f"\n- Adding more test cases to better understand failure patterns"
                    f"\n- Implementing progressive complexity handling"
                    f"\n- Adding better error recovery mechanisms"
                )
                
        # Check for declining trends
        declining = [s for s, stats in analysis['success_rate'].items() 
                    if stats['trend'] == 'declining']
        if declining:
            recommendations.append(
                f"Declining performance in strategies: {', '.join(declining)}. Consider:"
                f"\n- Reviewing recent changes that might have affected performance"
                f"\n- Adding regression tests for previously successful cases"
                f"\n- Implementing strategy refinement based on failure patterns"
            )
            
        analysis['recommendations'] = recommendations
