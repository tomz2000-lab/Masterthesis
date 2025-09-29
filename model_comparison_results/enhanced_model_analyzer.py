#!/usr/bin/env python3
"""
Enhanced Model-vs-Model Negotiation Analyzer
===========================================

Analyzes negotiation games to compare individual model performance,
extract per-round metrics, and visualize head-to-head model comparisons.
"""

import re
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime

@dataclass
class ModelPerformance:
    """Individual model performance in a single game"""
    model_name: str
    model_id: str  # model_a, model_b
    actual_model: str  # The actual model name like Llama-3.2-3B-Instruct
    role: str  # Employee, Manager, etc.
    utility_surplus: float
    risk_minimization: float
    deadline_sensitivity: float
    feasibility: float
    final_utility: float = 0.0

@dataclass  
class GameResult:
    """Complete result of a single negotiation game"""
    game_type: str
    iteration: int
    agreement_reached: bool
    agreement_round: int
    total_rounds: int
    model_a_performance: ModelPerformance
    model_b_performance: ModelPerformance
    winner: str  # 'model_a', 'model_b', or 'tie'
    utility_difference: float
    
    def __post_init__(self):
        """Calculate derived metrics"""
        self.utility_difference = abs(self.model_a_performance.utility_surplus - 
                                    self.model_b_performance.utility_surplus)
        
        # Determine winner based on utility surplus
        if self.model_a_performance.utility_surplus > self.model_b_performance.utility_surplus:
            self.winner = 'model_a'
        elif self.model_b_performance.utility_surplus > self.model_a_performance.utility_surplus:
            self.winner = 'model_b'
        else:
            self.winner = 'tie'

class ModelComparisonAnalyzer:
    """Enhanced analyzer for head-to-head model comparison"""
    
    def __init__(self, results_dir: str):
        self.results_dir = Path(results_dir)
        self.game_results: List[GameResult] = []
        
    def parse_output_file(self, file_path: str, game_type: str) -> List[GameResult]:
        """Parse output file and extract individual model performances"""
        results = []
        
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        # Split by iterations
        iteration_pattern = r'=== Iteration (\d+)/\d+ ==='
        iterations = re.split(iteration_pattern, content)
        
        for i in range(1, len(iterations), 2):
            if i + 1 < len(iterations):
                iteration_num = int(iterations[i])
                iteration_content = iterations[i + 1]
                
                result = self._extract_game_result(iteration_content, game_type, iteration_num)
                if result:
                    results.append(result)
        
        return results
    
    def _extract_game_result(self, content: str, game_type: str, iteration: int) -> Optional[GameResult]:
        """Extract complete game result from iteration content"""
        try:
            # Extract model configurations
            model_configs = self._extract_model_configs(content)
            
            # Extract agreement information
            agreement_match = re.search(r'Agreement reached: (True|False)', content)
            agreement_reached = agreement_match and agreement_match.group(1) == 'True'
            
            round_match = re.search(r'Agreement round: (\d+)', content)
            agreement_round = int(round_match.group(1)) if round_match else 0
            
            # Extract metrics
            metrics_match = re.search(r"Metrics: ({.*?})", content, re.DOTALL)
            if not metrics_match:
                return None
                
            try:
                metrics_str = metrics_match.group(1)
                # Fix common JSON issues
                metrics_str = metrics_str.replace("'", '"')
                metrics = json.loads(metrics_str)
            except json.JSONDecodeError:
                # Fallback: extract metrics with regex
                metrics = self._extract_metrics_regex(content)
            
            # Create model performances
            model_a_perf = ModelPerformance(
                model_name=model_configs.get('model_a', 'Unknown'),
                model_id='model_a',
                actual_model=model_configs.get('model_a', 'Unknown'),
                role=self._get_role(game_type, 'model_a'),
                utility_surplus=metrics.get('utility_surplus', {}).get('model_a', 0.0),
                risk_minimization=metrics.get('risk_minimization', {}).get('model_a', 0.0),
                deadline_sensitivity=metrics.get('deadline_sensitivity', {}).get('model_a', 0.0),
                feasibility=metrics.get('feasibility', {}).get('model_a', 0.0)
            )
            
            model_b_perf = ModelPerformance(
                model_name=model_configs.get('model_b', 'Unknown'),
                model_id='model_b', 
                actual_model=model_configs.get('model_b', 'Unknown'),
                role=self._get_role(game_type, 'model_b'),
                utility_surplus=metrics.get('utility_surplus', {}).get('model_b', 0.0),
                risk_minimization=metrics.get('risk_minimization', {}).get('model_b', 0.0),
                deadline_sensitivity=metrics.get('deadline_sensitivity', {}).get('model_b', 0.0),
                feasibility=metrics.get('feasibility', {}).get('model_b', 0.0)
            )
            
            return GameResult(
                game_type=game_type,
                iteration=iteration,
                agreement_reached=agreement_reached,
                agreement_round=agreement_round,
                total_rounds=self._count_rounds(content),
                model_a_performance=model_a_perf,
                model_b_performance=model_b_perf,
                winner='',  # Will be calculated in __post_init__
                utility_difference=0.0  # Will be calculated in __post_init__
            )
            
        except Exception as e:
            print(f"Error parsing iteration {iteration} for {game_type}: {e}")
            return None
    
    def _extract_model_configs(self, content: str) -> Dict[str, str]:
        """Extract model configuration information"""
        configs = {}
        
        # Look for model loading patterns
        model_patterns = [
            (r"model_a.*?meta-llama/([\w\.-]+)", 'model_a'),
            (r"model_b.*?meta-llama/([\w\.-]+)", 'model_b')
        ]
        
        for pattern, model_id in model_patterns:
            match = re.search(pattern, content)
            if match:
                configs[model_id] = match.group(1)
        
        return configs
    
    def _extract_metrics_regex(self, content: str) -> Dict[str, Dict[str, float]]:
        """Fallback method to extract metrics using regex"""
        metrics = {'utility_surplus': {}, 'risk_minimization': {}, 
                  'deadline_sensitivity': {}, 'feasibility': {}}
        
        # Extract utility surplus
        utility_match = re.search(r"utility_surplus.*?model_a['\"]:\s*([\d\.]+).*?model_b['\"]:\s*([\d\.]+)", content)
        if utility_match:
            metrics['utility_surplus']['model_a'] = float(utility_match.group(1))
            metrics['utility_surplus']['model_b'] = float(utility_match.group(2))
        
        return metrics
    
    def _get_role(self, game_type: str, model_id: str) -> str:
        """Get the role name for a model in a specific game"""
        role_mapping = {
            'company_car': {'model_a': 'Employee', 'model_b': 'Manager'},
            'resource_allocation': {'model_a': 'Development', 'model_b': 'Marketing'},
            'integrative_negotiation': {'model_a': 'IT Team', 'model_b': 'Marketing Team'}
        }
        return role_mapping.get(game_type, {}).get(model_id, 'Unknown')
    
    def _count_rounds(self, content: str) -> int:
        """Count the number of negotiation rounds"""
        rounds = len(re.findall(r'Round \d+|Turn \d+', content))
        if rounds == 0:
            rounds = len(re.findall(r'Player \w+ says:|Action:', content))
        return max(rounds, 1)
    
    def analyze_batch_run(self, batch_dir: str) -> None:
        """Analyze all games in a batch run"""
        batch_path = self.results_dir / batch_dir
        
        game_files = {
            'company_car': 'company_car_output.txt',
            'resource_allocation': 'resource_allocation_output.txt',
            'integrative_negotiation': 'integrative_negotiation_output.txt'
        }
        
        for game_type, filename in game_files.items():
            file_path = batch_path / filename
            if file_path.exists():
                results = self.parse_output_file(str(file_path), game_type)
                self.game_results.extend(results)
                print(f"Parsed {len(results)} games for {game_type}")
    
    def generate_model_comparison_report(self) -> str:
        """Generate detailed model-vs-model comparison report"""
        if not self.game_results:
            return "No game results to analyze."
        
        report = []
        report.append("MODEL-VS-MODEL COMPARISON ANALYSIS")
        report.append("=" * 50)
        
        # Overall model performance
        model_a_wins = sum(1 for r in self.game_results if r.winner == 'model_a')
        model_b_wins = sum(1 for r in self.game_results if r.winner == 'model_b')
        ties = sum(1 for r in self.game_results if r.winner == 'tie')
        
        report.append(f"\nOVERALL HEAD-TO-HEAD RESULTS:")
        report.append(f"Model A (3B) wins: {model_a_wins}/{len(self.game_results)} ({model_a_wins/len(self.game_results):.1%})")
        report.append(f"Model B (8B) wins: {model_b_wins}/{len(self.game_results)} ({model_b_wins/len(self.game_results):.1%})")
        report.append(f"Ties: {ties}/{len(self.game_results)} ({ties/len(self.game_results):.1%})")
        
        # Performance by game type
        report.append(f"\nPERFORMANCE BY GAME TYPE:")
        for game_type in ['company_car', 'resource_allocation', 'integrative_negotiation']:
            game_results = [r for r in self.game_results if r.game_type == game_type]
            if game_results:
                a_wins = sum(1 for r in game_results if r.winner == 'model_a')
                b_wins = sum(1 for r in game_results if r.winner == 'model_b')
                ties = sum(1 for r in game_results if r.winner == 'tie')
                
                avg_rounds = np.mean([r.agreement_round for r in game_results if r.agreement_reached])
                
                report.append(f"\n{game_type.upper()}:")
                report.append(f"  Model A (3B) wins: {a_wins}/{len(game_results)}")
                report.append(f"  Model B (8B) wins: {b_wins}/{len(game_results)}")
                report.append(f"  Average agreement round: {avg_rounds:.1f}")
                
                # Average utilities
                avg_a_utility = np.mean([r.model_a_performance.utility_surplus for r in game_results])
                avg_b_utility = np.mean([r.model_b_performance.utility_surplus for r in game_results])
                report.append(f"  Avg Model A utility: {avg_a_utility:.1f}")
                report.append(f"  Avg Model B utility: {avg_b_utility:.1f}")
        
        # Agreement round analysis
        agreement_rounds = [r.agreement_round for r in self.game_results if r.agreement_reached]
        if agreement_rounds:
            report.append(f"\nAGREEMENT TIMING ANALYSIS:")
            report.append(f"Average agreement round: {np.mean(agreement_rounds):.1f}")
            report.append(f"Median agreement round: {np.median(agreement_rounds):.1f}")
            report.append(f"Agreement round range: {min(agreement_rounds)} - {max(agreement_rounds)}")
        
        return "\n".join(report)
    
    def create_model_comparison_visualizations(self, save_dir: str = None):
        """Create comprehensive model comparison visualizations"""
        if not self.game_results:
            print("No results to visualize.")
            return
        
        # Create DataFrame for analysis
        data = []
        for result in self.game_results:
            # Model A row
            data.append({
                'game_type': result.game_type,
                'iteration': result.iteration,
                'model': 'Model_A_3B',
                'model_id': 'model_a',
                'role': result.model_a_performance.role,
                'utility_surplus': result.model_a_performance.utility_surplus,
                'risk_minimization': result.model_a_performance.risk_minimization,
                'feasibility': result.model_a_performance.feasibility,
                'agreement_round': result.agreement_round,
                'total_rounds': result.total_rounds,
                'won': result.winner == 'model_a'
            })
            
            # Model B row  
            data.append({
                'game_type': result.game_type,
                'iteration': result.iteration,
                'model': 'Model_B_8B',
                'model_id': 'model_b',
                'role': result.model_b_performance.role,
                'utility_surplus': result.model_b_performance.utility_surplus,
                'risk_minimization': result.model_b_performance.risk_minimization,
                'feasibility': result.model_b_performance.feasibility,
                'agreement_round': result.agreement_round,
                'total_rounds': result.total_rounds,
                'won': result.winner == 'model_b'
            })
        
        df = pd.DataFrame(data)
        
        # Create visualizations
        fig, axes = plt.subplots(3, 2, figsize=(16, 18))
        fig.suptitle('Model-vs-Model Negotiation Performance Analysis', fontsize=16, fontweight='bold')
        
        # 1. Win Rate by Model
        win_rates = df.groupby('model')['won'].mean()
        axes[0, 0].bar(win_rates.index, win_rates.values, color=['lightblue', 'lightcoral'])
        axes[0, 0].set_title('Overall Win Rate by Model')
        axes[0, 0].set_ylabel('Win Rate')
        axes[0, 0].set_ylim(0, 1)
        for i, v in enumerate(win_rates.values):
            axes[0, 0].text(i, v + 0.02, f'{v:.1%}', ha='center')
        
        # 2. Utility Surplus Comparison
        sns.boxplot(data=df, x='game_type', y='utility_surplus', hue='model', ax=axes[0, 1])
        axes[0, 1].set_title('Utility Surplus by Game Type and Model')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # 3. Agreement Round Distribution
        agreement_data = df[df['agreement_round'] > 0]
        sns.histplot(data=agreement_data, x='agreement_round', hue='model', 
                    multiple='dodge', bins=range(1, max(agreement_data['agreement_round'])+2), ax=axes[1, 0])
        axes[1, 0].set_title('Agreement Round Distribution')
        axes[1, 0].set_xlabel('Round When Agreement Reached')
        
        # 4. Performance by Game Type (Win Rates)
        game_wins = df.groupby(['game_type', 'model'])['won'].mean().unstack()
        game_wins.plot(kind='bar', ax=axes[1, 1])
        axes[1, 1].set_title('Win Rate by Game Type')
        axes[1, 1].set_ylabel('Win Rate')
        axes[1, 1].tick_params(axis='x', rotation=45)
        axes[1, 1].legend(title='Model')
        
        # 5. Utility vs Agreement Round
        sns.scatterplot(data=agreement_data, x='agreement_round', y='utility_surplus', 
                       hue='model', style='game_type', size='feasibility', ax=axes[2, 0])
        axes[2, 0].set_title('Utility vs Agreement Timing')
        axes[2, 0].set_xlabel('Agreement Round')
        axes[2, 0].set_ylabel('Utility Surplus')
        
        # 6. Head-to-Head Summary
        # Create a summary table visualization
        game_summary = []
        for game_type in df['game_type'].unique():
            game_df = df[df['game_type'] == game_type]
            model_a_wins = game_df[(game_df['model'] == 'Model_A_3B') & (game_df['won'])].shape[0]
            model_b_wins = game_df[(game_df['model'] == 'Model_B_8B') & (game_df['won'])].shape[0]
            game_summary.append([game_type, model_a_wins, model_b_wins])
        
        axes[2, 1].axis('tight')
        axes[2, 1].axis('off')
        table_data = [['Game Type', 'Model A (3B) Wins', 'Model B (8B) Wins']] + game_summary
        table = axes[2, 1].table(cellText=table_data, cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        axes[2, 1].set_title('Head-to-Head Summary')
        
        plt.tight_layout()
        
        if save_dir:
            save_path = Path(save_dir)
            save_path.mkdir(exist_ok=True)
            fig.savefig(save_path / 'model_comparison_analysis.png', dpi=300, bbox_inches='tight')
            print(f"Model comparison visualization saved to {save_path / 'model_comparison_analysis.png'}")
        
        plt.show()
        return fig, df

def main():
    """Main analysis function"""
    analyzer = ModelComparisonAnalyzer('.')
    
    print("=" * 60)
    print("ENHANCED MODEL-VS-MODEL ANALYSIS")
    print("=" * 60)
    
    # Analyze batch run
    analyzer.analyze_batch_run('batch_run_1')
    
    if not analyzer.game_results:
        print("No game results found!")
        return
    
    print(f"\nAnalyzed {len(analyzer.game_results)} total negotiations")
    
    # Generate report
    report = analyzer.generate_model_comparison_report()
    print("\n" + report)
    
    # Save detailed report
    with open('model_comparison_detailed_report.txt', 'w') as f:
        f.write(report)
    
    # Create detailed CSV
    detailed_data = []
    for result in analyzer.game_results:
        detailed_data.append({
            'game_type': result.game_type,
            'iteration': result.iteration,
            'agreement_reached': result.agreement_reached,
            'agreement_round': result.agreement_round,
            'winner': result.winner,
            'model_a_utility': result.model_a_performance.utility_surplus,
            'model_b_utility': result.model_b_performance.utility_surplus,
            'utility_difference': result.utility_difference,
            'model_a_risk_min': result.model_a_performance.risk_minimization,
            'model_b_risk_min': result.model_b_performance.risk_minimization,
            'model_a_feasibility': result.model_a_performance.feasibility,
            'model_b_feasibility': result.model_b_performance.feasibility
        })
    
    pd.DataFrame(detailed_data).to_csv('model_comparison_detailed_data.csv', index=False)
    
    # Create visualizations
    fig, df = analyzer.create_model_comparison_visualizations('.')
    
    print("\n" + "=" * 60)
    print("FILES GENERATED:")
    print("- model_comparison_detailed_report.txt")
    print("- model_comparison_detailed_data.csv") 
    print("- model_comparison_analysis.png")
    print("=" * 60)
    
    return analyzer

if __name__ == "__main__":
    analyzer = main()