"""Analysis utilities for test results"""
from typing import Dict, List, Any
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
from pathlib import Path

class ResultAnalyzer:
    def __init__(self, results: Dict[str, Any]):
        self.results = results
        self.strategy_data = defaultdict(list)
        self._process_results()

    def _process_results(self):
        """Process raw results into analyzable data"""
        for strategy, attempts in self.results['by_strategy'].items():
            self.strategy_data[strategy] = {
                'success_rates': [],
                'times': [],
                'errors': defaultdict(int)
            }
            
            for attempt in attempts:
                self.strategy_data[strategy]['success_rates'].append(int(attempt['success']))
                self.strategy_data[strategy]['times'].append(attempt['time_taken'])
                if not attempt['success'] and attempt.get('error'):
                    self.strategy_data[strategy]['errors'][attempt['error']] += 1

    def generate_plots(self, output_dir: str = 'analysis'):
        """Generate analysis plots"""
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)

        self._plot_success_rates(output_dir / 'success_rates.png')
        self._plot_timing_distribution(output_dir / 'timing_dist.png')
        self._plot_error_distribution(output_dir / 'error_dist.png')

    def _plot_success_rates(self, output_path: Path):
        """Plot success rates by strategy"""
        strategies = list(self.strategy_data.keys())
        success_rates = [np.mean(self.strategy_data[s]['success_rates']) 
                        for s in strategies]

        plt.figure(figsize=(10, 6))
        plt.bar(strategies, success_rates)
        plt.title('Success Rates by Strategy')
        plt.ylabel('Success Rate')
        plt.ylim(0, 1)
        
        for i, v in enumerate(success_rates):
            plt.text(i, v + 0.01, f'{v:.2%}', ha='center')
            
        plt.savefig(output_path)
        plt.close()

    def _plot_timing_distribution(self, output_path: Path):
        """Plot timing distribution by strategy"""
        plt.figure(figsize=(12, 6))
        
        positions = []
        times = []
        labels = []
        
        for i, (strategy, data) in enumerate(self.strategy_data.items()):
            positions.append(i)
            times.append(data['times'])
            labels.append(strategy)
            
        plt.boxplot(times, positions=positions, labels=labels)
        plt.title('Solving Time Distribution by Strategy')
        plt.ylabel('Time (seconds)')
        plt.yscale('log')
        
        plt.savefig(output_path)
        plt.close()

    def _plot_error_distribution(self, output_path: Path):
        """Plot error distribution by strategy"""
        plt.figure(figsize=(15, 8))
        
        strategies = list(self.strategy_data.keys())
        error_types = set()
        for data in self.strategy_data.values():
            error_types.update(data['errors'].keys())
        error_types = list(error_types)
        
        data = []
        for strategy in strategies:
            strategy_errors = self.strategy_data[strategy]['errors']
            data.append([strategy_errors[error] for error in error_types])
            
        data = np.array(data)
        
        bottom = np.zeros(len(strategies))
        for i, error in enumerate(error_types):
            plt.bar(strategies, data[:, i], bottom=bottom, label=error)
            bottom += data[:, i]
            
        plt.title('Error Distribution by Strategy')
        plt.ylabel('Number of Errors')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        
        plt.savefig(output_path)
        plt.close()

    def generate_correlations(self) -> Dict[str, float]:
        """Generate correlation analysis"""
        correlations = {}
        
        # Correlate success with various factors
        for strategy, data in self.strategy_data.items():
            if len(data['success_rates']) > 1 and len(data['times']) > 1:
                corr = np.corrcoef(data['success_rates'], data['times'])[0, 1]
                correlations[f'{strategy}_time_vs_success'] = corr
                
        return correlations

    def identify_patterns(self) -> Dict[str, List[str]]:
        """Identify patterns in the results"""
        patterns = defaultdict(list)
        
        # Time-based patterns
        for strategy, data in self.strategy_data.items():
            mean_time = np.mean(data['times'])
            if mean_time > 5.0:
                patterns['slow_strategies'].append(strategy)
            
            time_variance = np.var(data['times'])
            if time_variance > 10.0:
                patterns['inconsistent_timing'].append(strategy)
                
        # Success patterns
        for strategy, data in self.strategy_data.items():
            success_rate = np.mean(data['success_rates'])
            if success_rate < 0.5:
                patterns['low_success'].append(strategy)
            
        # Error patterns
        for strategy, data in self.strategy_data.items():
            if len(data['errors']) > 3:
                patterns['diverse_errors'].append(strategy)
                
        return dict(patterns)

    def generate_insights(self) -> List[str]:
        """Generate insights from the analysis"""
        insights = []
        correlations = self.generate_correlations()
        patterns = self.identify_patterns()
        
        # Correlation insights
        for metric, corr in correlations.items():
            if abs(corr) > 0.7:
                insights.append(f"Strong correlation ({corr:.2f}) found in {metric}")
                
        # Pattern insights
        for pattern_type, strategies in patterns.items():
            if strategies:
                insights.append(f"{pattern_type}: {', '.join(strategies)}")
                
        # Performance insights
        for strategy, data in self.strategy_data.items():
            success_rate = np.mean(data['success_rates'])
            mean_time = np.mean(data['times'])
            insights.append(
                f"{strategy}: {success_rate:.2%} success rate, "
                f"{mean_time:.2f}s average time"
            )
            
        return insights