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
                'by_strategy': results,  # Store full results
                'analysis_dir': str(analysis_dir),
                'status': 'completed'
            }
            
            # Track improvements/regressions if we have previous runs
            if self.history['runs']:
                last_run = self.history['runs'][-1]
                if 'by_strategy' in last_run:
                    improvements = self._analyze_improvements(
                        last_run['by_strategy'], 
                        results
                    )
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
        
        try:
            # Compare success rates for each strategy
            for strategy in old_results:
                if strategy in new_results:
                    # Get success rates
                    old_successes = sum(1 for a in old_results[strategy] if a.get('success', False))
                    old_total = len(old_results[strategy])
                    old_rate = old_successes / old_total if old_total > 0 else 0

                    new_successes = sum(1 for a in new_results[strategy] if a.get('success', False))
                    new_total = len(new_results[strategy])
                    new_rate = new_successes / new_total if new_total > 0 else 0
                    
                    change = new_rate - old_rate
                    improvements['success_rate'][strategy] = {
                        'change': change,
                        'status': 'improved' if change > 0 else 'regressed' if change < 0 else 'unchanged'
                    }
                    
                    # Compare speeds
                    old_time = sum(a.get('time_taken', 0) for a in old_results[strategy]) / old_total
                    new_time = sum(a.get('time_taken', 0) for a in new_results[strategy]) / new_total
                    
                    time_change = old_time - new_time  # Positive means faster
                    improvements['speed'][strategy] = {
                        'change': time_change,
                        'status': 'improved' if time_change > 0 else 'regressed' if time_change < 0 else 'unchanged'
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
            
        except Exception as e:
            self.logger.error(f"Error analyzing improvements: {e}")
            return {
                'success_rate': {},
                'speed': {},
                'overall': 'error',
                'error': str(e)
            }