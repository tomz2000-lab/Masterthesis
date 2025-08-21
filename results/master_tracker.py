"""
Simple Master Tracker
====================
Combines all four metrics in one simple interface.
"""

import pandas as pd
from datetime import datetime
from typing import Dict, Any
import os

from .utility_surplus_tracker import SimpleUtilityTracker
from .risk_minimization_tracker import SimpleRiskTracker
from .deadline_sensitivity_tracker import SimpleDeadlineTracker
from .feasibility_tracker import SimpleFeasibilityTracker

class SimpleMasterTracker:
    """Master tracker for all four metrics: model × game × metrics"""
    
    def __init__(self, results_dir: str = "."):
        self.utility_tracker = SimpleUtilityTracker(results_dir)
        self.risk_tracker = SimpleRiskTracker(results_dir)
        self.deadline_tracker = SimpleDeadlineTracker(results_dir)
        self.feasibility_tracker = SimpleFeasibilityTracker(results_dir)
    
    def record_session(self, session_data: Dict[str, Any]):
        """Record all metrics from one session"""
        print(f"\n📊 Recording session: {session_data.get('game_type', 'unknown')}")
        
        # Record each metric
        self.utility_tracker.record_result(session_data)
        self.risk_tracker.record_result(session_data)
        self.deadline_tracker.record_result(session_data)
        self.feasibility_tracker.record_result(session_data)
    
    def get_combined_summary(self) -> pd.DataFrame:
        """Get combined model × game performance across all metrics"""
        
        # Load all data
        utility_df = self.utility_tracker.load_data()
        risk_df = self.risk_tracker.load_data()
        deadline_df = self.deadline_tracker.load_data()
        feasibility_df = self.feasibility_tracker.load_data()
        
        if all(df.empty for df in [utility_df, risk_df, deadline_df, feasibility_df]):
            return pd.DataFrame()
        
        # Create summary tables for each metric
        summaries = {}
        
        if not utility_df.empty:
            summaries['Utility_Surplus'] = utility_df.groupby(['model_id', 'game_type'])['utility_surplus'].mean().unstack(fill_value=0)
        
        if not risk_df.empty:
            summaries['Risk_Percentage'] = risk_df.groupby(['model_id', 'game_type'])['risk_percentage'].mean().unstack(fill_value=0)
        
        if not deadline_df.empty:
            summaries['Deadline_Sensitivity'] = deadline_df.groupby(['model_id', 'game_type'])['deadline_sensitivity'].mean().unstack(fill_value=0)
        
        if not feasibility_df.empty:
            summaries['Feasibility_Score'] = feasibility_df.groupby(['model_id', 'game_type'])['feasibility_score'].mean().unstack(fill_value=0)
        
        return summaries
    
    def generate_master_report(self) -> str:
        """Generate comprehensive but simple report"""
        summaries = self.get_combined_summary()
        
        if not summaries:
            return "No data available across any metrics"
        
        report = f"""
=== COMPLETE PERFORMANCE DASHBOARD ===
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

"""
        
        for metric_name, summary_table in summaries.items():
            report += f"\n{metric_name.upper()} BY MODEL & GAME:\n"
            report += f"{summary_table.to_string()}\n"
            report += "-" * 50 + "\n"
        
        return report
    
    def get_model_comparison(self) -> str:
        """Compare models across all metrics and games"""
        summaries = self.get_combined_summary()
        
        if not summaries:
            return "No data for comparison"
        
        comparison = f"""
=== MODEL COMPARISON ACROSS ALL METRICS ===

"""
        
        # Get all unique models and games
        all_models = set()
        all_games = set()
        
        for summary_table in summaries.values():
            all_models.update(summary_table.index)
            all_games.update(summary_table.columns)
        
        # Compare each model
        for model in sorted(all_models):
            comparison += f"\n{model.upper()}:\n"
            for metric_name, summary_table in summaries.items():
                if model in summary_table.index:
                    avg_performance = summary_table.loc[model].mean()
                    comparison += f"  {metric_name}: {avg_performance:.1f}\n"
            comparison += "\n"
        
        return comparison

# Example usage and testing
if __name__ == "__main__":
    tracker = SimpleMasterTracker()
    
    # Mock session
    mock_session = {
        'game_type': 'resource_allocation',
        'metrics': {
            'utility_surplus': {'model_a': 45.2, 'model_b': 32.8},
            'risk_minimization': {'model_a': 0.0, 'model_b': 0.0},
            'deadline_sensitivity': {'model_a': 100.0, 'model_b': 100.0},
            'feasibility': {'model_a': 95.0, 'model_b': 100.0}
        },
        'model_configs': {
            'model_a': {'config': {'temperature': 0.7}},
            'model_b': {'config': {'temperature': 0.5}}
        }
    }
    
    # Test
    # tracker.record_session(mock_session)
    # print(tracker.generate_master_report())
    # print(tracker.get_model_comparison())
